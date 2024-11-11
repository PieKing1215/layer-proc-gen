use ::rand::distributions::uniform::SampleRange as _;
use arrayvec::ArrayVec;
use debug::{DebugContent, DynLayer};
use generic_layers::{rng_for_point, ReducedUniformPoint, Reducible};
use macroquad::prelude::*;
use miniquad::window::screen_size;
use std::{
    borrow::Borrow,
    cell::{Cell, Ref, RefCell},
    collections::{BTreeMap, HashMap},
    f32::consts::{FRAC_PI_2, PI},
    num::NonZeroU8,
    ops::Range,
    sync::Arc,
};

use layer_proc_gen::*;
use rigid2d::Body;
use vec2::{Bounds, Line, Num, Point2d};

#[derive(PartialEq, Debug, Clone, Default)]
struct City {
    center: Point2d,
    size: i64,
    name: String,
}

impl From<Point2d> for City {
    fn from(center: Point2d) -> Self {
        let mut rng = rng_for_point::<0, _>(center);
        City {
            center,
            size: { Self::SIZES.sample_single(&mut rng) },
            name: (0..(3..12).sample_single(&mut rng))
                .map(|_| ('a'..='z').sample_single(&mut rng))
                .collect(),
        }
    }
}

impl City {
    const SIZES: Range<i64> = 100..500;
}

impl Reducible for City {
    fn radius(&self) -> i64 {
        self.size
    }

    fn position(&self) -> Point2d {
        self.center
    }

    fn debug(&self) -> Vec<DebugContent> {
        vec![
            DebugContent::Circle {
                center: self.center,
                radius: self.size as f32,
            },
            DebugContent::Text {
                pos: self.center,
                label: self.name.clone(),
            },
        ]
    }
}

#[derive(Clone, PartialEq, Default)]
struct Intersection(Point2d);

impl From<Point2d> for Intersection {
    fn from(value: Point2d) -> Self {
        Self(value)
    }
}

impl Reducible for Intersection {
    fn radius(&self) -> i64 {
        50
    }

    fn position(&self) -> Point2d {
        self.0
    }
}

/// Removes locations that are too close to others
#[derive(PartialEq, Debug, Clone, Default)]
struct ReducedLocations {
    points: ArrayVec<Point2d, 7>,
    trees: ArrayVec<Point2d, 7>,
}

impl Chunk for ReducedLocations {
    type LayerStore<T> = Arc<T>;
    type Dependencies = (
        ReducedUniformPoint<Intersection, 6, 0>,
        ReducedUniformPoint<City, 11, 1>,
    );
    const SIZE: Point2d<u8> = Point2d::splat(6);

    fn compute(
        (raw_locations, cities): &<Self::Dependencies as Dependencies>::Layer,
        index: GridPoint<Self>,
    ) -> Self {
        let bounds = Self::bounds(index);
        let center = bounds.center();
        let points = raw_locations
            .get_or_compute(index.into_same_chunk_size())
            .points
            .iter()
            .map(|p| p.0)
            .collect();
        if cities
            .get_range(Bounds::point(center).pad(Point2d::splat(City::SIZES.end)))
            .all(|cities| {
                cities
                    .points
                    .iter()
                    .all(|city| center.manhattan_dist(city.center) > city.size)
            })
        {
            ReducedLocations {
                points: ArrayVec::default(),
                trees: points,
            }
        } else {
            ReducedLocations {
                points,
                trees: ArrayVec::default(),
            }
        }
    }

    fn debug_contents(&self) -> Vec<DebugContent> {
        self.trees
            .iter()
            .map(|&center| DebugContent::Circle { center, radius: 8. })
            .chain(
                self.points
                    .iter()
                    .map(|&center| DebugContent::Circle { center, radius: 1. }),
            )
            .collect()
    }
}

#[derive(PartialEq, Debug, Default, Clone)]
struct Roads {
    roads: Arc<Vec<Line>>,
}

impl Chunk for Roads {
    type LayerStore<T> = T;
    type Dependencies = (ReducedLocations,);
    const SIZE: Point2d<u8> = Point2d::splat(6);

    fn compute(
        (locations,): &<Self::Dependencies as Dependencies>::Layer,
        index: GridPoint<Self>,
    ) -> Self {
        let roads = gen_roads(
            locations
                .get_moore_neighborhood(index.into_same_chunk_size())
                .map(|chunk| chunk.points),
            |&p| p,
            |&a, &b| a.to(b),
        )
        .into();
        Roads { roads }
    }

    fn debug_contents(&self) -> Vec<DebugContent> {
        self.roads.iter().copied().map(DebugContent::from).collect()
    }
}

fn gen_roads<T: Clone, U>(
    chunks: impl Iterator<Item = impl Borrow<[T]>>,
    get_point: impl Fn(&T) -> Point2d,
    mk: impl Fn(&T, &T) -> U,
) -> Vec<U> {
    let mut roads = vec![];
    let mut points: ArrayVec<T, { 3 * 9 }> = ArrayVec::new();
    let mut start = usize::MAX;
    let mut n = usize::MAX;
    for (i, grid) in chunks.enumerate() {
        let grid = grid.borrow();
        if i == 4 {
            start = points.len();
            n = grid.len();
        }
        points.extend(grid.iter().cloned());
    }
    // We only care about the roads starting from the center grid cell, as the others are not necessarily correct,
    // or will be computed by the other grid cells.
    // The others may connect the outer edges of the current grid range and thus connect roads that
    // don't satisfy the algorithm.
    // This algorithm is https://en.m.wikipedia.org/wiki/Relative_neighborhood_graph adjusted for
    // grid-based computation. It's a brute force implementation, but I think that is faster than going through
    // a Delaunay triangulation first, as instead of (3*9)^3 = 19683 inner loop iterations we have only
    // 3 * (2 + 1 + 3*4) * 3*9 = 1215
    // FIXME: cache distance computations as we do them, we can save 1215-(3*9^3)/2 = 850 distance computations (70%) and figure
    // out how to cache them across grid cells (along with removing them from the cache when they aren't needed anymore)
    // as the neighboring cells will be redoing the same distance computations.
    for (i, a_val) in points.iter().enumerate().skip(start).take(n) {
        let a = get_point(a_val);
        for b_val in points.iter().skip(i + 1) {
            let b = get_point(b_val);
            let dist = a.dist_squared(b);
            if points.iter().all(|c| {
                let c = get_point(c);
                if a == c || b == c {
                    return true;
                }
                // FIXME: make cheaper by already bailing if `x*x` is larger than dist,
                // to avoid computing `y*y`.
                let a_dist = a.dist_squared(c);
                let b_dist = c.dist_squared(b);
                dist < a_dist || dist < b_dist
            }) {
                roads.push(mk(a_val, b_val))
            }
        }
    }
    roads
}

#[derive(PartialEq, Debug, Clone)]

struct Highway {
    line: Line,
    start_city: String,
    start_sign: String,
    end_city: String,
    end_sign: String,
}

#[derive(PartialEq, Debug, Default, Clone)]
struct Highways {
    roads: Arc<Vec<Highway>>,
}

impl Chunk for Highways {
    type LayerStore<T> = T;
    type Dependencies = (ReducedUniformPoint<City, 11, 1>, ReducedLocations);
    const SIZE: Point2d<u8> = ReducedUniformPoint::<City, 11, 1>::SIZE;

    fn compute(
        (cities, locations): &<Self::Dependencies as Dependencies>::Layer,
        index: GridPoint<Self>,
    ) -> Self {
        let roads = gen_roads(
            cities
                .get_moore_neighborhood(index.into_same_chunk_size())
                .map(|chunk| chunk.points),
            |p| p.center,
            |a, b| {
                (
                    a.size,
                    b.size,
                    a.center.to(b.center),
                    a.name.clone(),
                    b.name.clone(),
                )
            },
        );

        let roads = roads
            .into_iter()
            .map(|(start_size, end_size, road, start_city, end_city)| {
                let approx_start = road.with_manhattan_length(start_size).end;
                let approx_end = road.flip().with_manhattan_length(end_size).end;

                let closest = |p, start| {
                    let mut closest = None;
                    Chunk::pos_to_grid(p)
                        .to(Chunk::pos_to_grid(start))
                        .iter_all_touched_pixels(|index| {
                            closest = locations
                                .get_or_compute(index)
                                .points
                                .iter()
                                .copied()
                                .chain(closest)
                                .min_by_key(|point| point.dist_squared(p))
                        });
                    closest
                };
                let line = Line {
                    start: closest(approx_start, road.start).unwrap_or(approx_start),
                    end: closest(approx_end, road.end).unwrap_or(approx_end),
                };
                let dist_km = ((line.len_squared() as f32).sqrt() / 1000.).ceil();
                Highway {
                    line,
                    start_sign: format!("{end_city} {dist_km}km"),
                    end_sign: format!("{start_city} {dist_km}km"),
                    start_city,
                    end_city,
                }
            })
            .collect();
        Highways {
            roads: Arc::new(roads),
        }
    }

    fn debug_contents(&self) -> Vec<DebugContent> {
        self.roads
            .iter()
            .map(|highway| DebugContent::Line(highway.line))
            .collect()
    }
}

struct Player {
    roads: Layer<Roads>,
    trees: Layer<ReducedLocations>,
    highways: Layer<Highways>,
    max_zoom_in: NonZeroU8,
    max_zoom_out: NonZeroU8,
    car: Car,
    last_grid_vision_range: Cell<(Bounds<GridIndex<Roads>>, Bounds<GridIndex<Highways>>)>,
    roads_for_last_grid_vision_range: RefCell<(Vec<Highway>, Vec<Tree>)>,
}

struct Tree {
    pos: Point2d,
}

impl Player {
    pub fn new(
        roads: Layer<Roads>,
        highways: Layer<Highways>,
        trees: Layer<ReducedLocations>,
    ) -> Self {
        Self {
            roads: roads.into(),
            highways: highways.into(),
            trees,
            max_zoom_in: NonZeroU8::new(5).unwrap(),
            max_zoom_out: NonZeroU8::new(10).unwrap(),
            car: Car {
                length: 4.,
                width: 2.,
                body: Default::default(),
                steering_limit: 15,
                steering: 0.,
                color: DARKPURPLE,
                braking: false,
                reversing: false,
            },
            last_grid_vision_range: (
                Bounds::point(Point2d::splat(GridIndex::from_raw(0))),
                Bounds::point(Point2d::splat(GridIndex::from_raw(0))).into(),
            )
                .into(),
            roads_for_last_grid_vision_range: (vec![], vec![]).into(),
        }
    }

    /// Absolute position and function to go from a global position
    /// to one relative to the player.
    pub fn point2screen(&self) -> impl Fn(Point2d) -> Vec2 {
        let player_pos = self.pos();

        // Avoid moving everything in whole pixels and allow for smooth sub-pixel movement instead
        let adjust = self.car.body.position.fract();
        move |point: Point2d| -> Vec2 {
            let point = point - player_pos;
            i64vec2(point.x, point.y).as_vec2() - adjust
        }
    }

    fn pos(&self) -> Point2d {
        Point2d {
            x: self.car.body.position.x as i64,
            y: self.car.body.position.y as i64,
        }
    }

    pub fn vision_range<C: Chunk>(&self, half_screen_visible_area: Vec2) -> Bounds {
        let padding = half_screen_visible_area.abs().ceil().as_i64vec2();
        Bounds::point(self.pos())
            // pad by the screen area, so everything that will get rendered is within the vision range
            .pad(Point2d::new(padding.x, padding.y))
            // Pad by a chunk size to make sure we see effects from the neighboring chunks
            .pad(C::SIZE.map(|i| 1 << i))
    }

    pub fn grid_vision_range<C: Chunk>(
        &self,
        half_screen_visible_area: Vec2,
    ) -> Bounds<GridIndex<C>> {
        C::bounds_to_grid(self.vision_range::<C>(half_screen_visible_area))
    }

    pub fn roads(&self, half_screen_visible_area: Vec2) -> Ref<'_, (Vec<Highway>, Vec<Tree>)> {
        let grid_vision_range = self.grid_vision_range(half_screen_visible_area);
        let highway_vision_range = self.grid_vision_range(half_screen_visible_area);
        if (grid_vision_range, highway_vision_range) != self.last_grid_vision_range.get() {
            self.last_grid_vision_range
                .set((grid_vision_range, highway_vision_range));
            let mut roads = self.roads_for_last_grid_vision_range.borrow_mut();
            let (roads, trees) = &mut *roads;
            roads.clear();
            trees.clear();
            for index in grid_vision_range.iter() {
                for &line in self.roads.get_or_compute(index).roads.iter() {
                    roads.push(Highway {
                        line,
                        start_city: String::new(),
                        start_sign: String::new(),
                        end_city: String::new(),
                        end_sign: String::new(),
                    });
                }
            }
            for index in highway_vision_range.iter() {
                roads.extend_from_slice(&self.highways.get_or_compute(index).roads);
            }
            for index in grid_vision_range.iter() {
                for &tree in &self
                    .trees
                    .get_or_compute(index.into_same_chunk_size())
                    .trees
                {
                    trees.push(Tree { pos: tree });
                }
            }
        }
        self.roads_for_last_grid_vision_range.borrow()
    }
}

#[macroquad::main("layer proc gen demo")]
async fn main() {
    let mut camera = Camera2D::default();
    let standard_zoom = Vec2::from(screen_size()).recip() * 4.;
    camera.zoom = standard_zoom;
    set_camera(&camera);
    let mut overlay_camera = Camera2D::default();
    overlay_camera.zoom = standard_zoom / 4.;
    overlay_camera.offset = vec2(-1., 1.);

    let cities = Layer::new(ReducedUniformPoint::default_layer());
    let locations = Layer::new((Default::default(), cities.clone()));
    let mut player = Player::new(
        Layer::new((locations.clone(),)),
        Layer::new((cities.clone(), locations.clone())),
        locations.clone(),
    );

    let start_city = cities
        .get_grid_range(
            Bounds::point(Point2d::splat(GridIndex::ZERO)).pad(Point2d::splat(GridIndex::TWO)),
        )
        .flat_map(|c| c.points.into_iter())
        .next()
        .expect("you wont the lottery, no cities in a 5x5 grid");
    let start_road = player
        .roads
        .get_range(Bounds::point(start_city.center).pad(Point2d::splat(start_city.size)))
        .find_map(|c| c.roads.iter().copied().next())
        .expect("you wont the lottery, no roads in a city");
    player.car.body.position = vec2(start_road.start.x as f32, start_road.start.y as f32);
    let dir = start_road.end - start_road.end;
    player.car.body.rotation = vec2(dir.x as f32, dir.y as f32).to_angle() + FRAC_PI_2;

    let mut smooth_cam_speed = 0.0;
    let mut debug_zoom = 1.0;
    let mut debug_view = false;
    let mut debug_chunks = false;

    loop {
        if is_key_pressed(KeyCode::Escape) {
            return;
        }
        if is_key_pressed(KeyCode::F3) {
            render_debug_layers(vec![
                locations.debug(),
                player.highways.debug(),
                player.roads.debug(),
            ])
            .await;
        }
        if is_key_pressed(KeyCode::F4) {
            render_3d_layers(vec![
                locations.debug(),
                player.highways.debug(),
                player.roads.debug(),
            ])
            .await;
        }
        player.car.update(Actions {
            accelerate: is_key_down(KeyCode::W),
            reverse: is_key_down(KeyCode::S),
            hand_brake: is_key_down(KeyCode::Space),
            left: is_key_down(KeyCode::A),
            right: is_key_down(KeyCode::D),
        });
        if is_key_pressed(KeyCode::Up) {
            debug_zoom *= 2.0;
        }
        if is_key_pressed(KeyCode::Down) {
            debug_zoom /= 2.0;
        }
        if is_key_pressed(KeyCode::F1) {
            debug_view = !debug_view;
        }
        if is_key_pressed(KeyCode::F2) {
            debug_chunks = !debug_chunks;
        }

        smooth_cam_speed = smooth_cam_speed * 0.99 + player.car.body.velocity.length() / 30. * 0.01;
        let max_zoom_in = f32::from(player.max_zoom_in.get());
        let max_zoom_out = f32::from(player.max_zoom_out.get());
        smooth_cam_speed = smooth_cam_speed.clamp(0.0, max_zoom_in);
        camera.zoom = standard_zoom * (max_zoom_in + 1.0 / max_zoom_out - smooth_cam_speed);
        camera.zoom /= debug_zoom;
        set_camera(&camera);
        camera.zoom *= debug_zoom;

        let point2screen = player.point2screen();
        clear_background(DARKGREEN);

        let draw_bounds = |bounds: Bounds, color| {
            if !debug_view {
                return;
            }
            let min = point2screen(bounds.min);
            let max = point2screen(bounds.max);
            draw_rectangle_lines(
                min.x as f32,
                min.y as f32,
                (max.x - min.x) as f32,
                (max.y - min.y) as f32,
                debug_zoom,
                color,
            );
        };

        let padding = camera.screen_to_world(Vec2::splat(0.));

        let draw_line = |line: Line, thickness, color| {
            let start = point2screen(line.start);
            let end = point2screen(line.end);
            draw_line(start.x, start.y, end.x, end.y, thickness, color);
        };

        let roads = player.roads(padding);
        let (roads, trees) = &*roads;
        for highway in roads.iter() {
            let start = point2screen(highway.line.start);
            let end = point2screen(highway.line.end);
            draw_line(highway.line, 8., GRAY);
            draw_circle(start.x, start.y, 4., GRAY);
            draw_circle(start.x, start.y, 0.1, WHITE);
            draw_circle(end.x, end.y, 4., GRAY);
            draw_circle(end.x, end.y, 0.1, WHITE);
            for (start, end, sign, name) in [
                (start, end, &highway.start_sign, &highway.start_city),
                (end, start, &highway.end_sign, &highway.end_city),
            ] {
                if sign.is_empty() && name.is_empty() {
                    continue;
                }
                let direction = end - start;
                let mut rotation = direction.to_angle();
                let mut sign_offset = 6. * 0.2;
                let mut name_offset = 14. * 0.2;
                let mut sign_line_distance = -1.;
                if rotation.abs() < PI / 2. {
                    std::mem::swap(&mut sign_offset, &mut name_offset);
                    sign_line_distance *= -1.;
                }
                if rotation > PI / 2. {
                    rotation -= PI;
                } else if rotation < -PI / 2. {
                    rotation += PI;
                }
                let pos = start
                    + direction.perp().normalize() * (sign_offset + 4.)
                    + direction.normalize() * 100.;
                draw_multiline_text_ex(
                    sign,
                    pos.x,
                    pos.y,
                    // bug in macroquad: line distance factor is applied on y axis, ignoring rotation
                    Some(sign_line_distance),
                    TextParams {
                        font_size: 20,
                        font_scale: 0.2,
                        rotation,
                        color: WHITE,
                        ..Default::default()
                    },
                );
                let pos = start - direction.perp().normalize() * name_offset
                    + direction.normalize() * 50.;
                draw_text_ex(
                    name,
                    pos.x,
                    pos.y,
                    TextParams {
                        font_size: 20,
                        font_scale: 0.2,
                        rotation,
                        color: WHITE,
                        ..Default::default()
                    },
                );
            }
        }
        for highway in roads.iter() {
            draw_line(highway.line, 0.2, WHITE);
        }

        for tree in trees.iter() {
            let pos = point2screen(tree.pos);
            draw_circle(
                pos.x,
                pos.y,
                8.,
                Color {
                    r: 0.0,
                    g: 0.30,
                    b: 0.05,
                    a: 1.0,
                },
            );
        }

        player.car.draw();

        let draw_debug_content = |debug: DebugContent, thickness, color| match debug {
            DebugContent::Line(line) => draw_line(line, thickness, color),
            DebugContent::Circle { center, radius } => {
                let pos = point2screen(center);
                draw_circle_lines(pos.x, pos.y, radius, thickness, color)
            }
            DebugContent::Text { pos, label } => {
                let pos = point2screen(pos);
                draw_multiline_text(&label, pos.x, pos.y, 100., Some(1.), color);
            }
        };
        let draw_layer_debug = |layer: &dyn DynLayer, color| {
            for (current_chunk, chunk) in layer.iter_all_loaded() {
                draw_bounds(current_chunk, color);
                for debug in chunk.render() {
                    draw_debug_content(debug, debug_zoom, color)
                }
            }
        };

        if debug_chunks {
            draw_layer_debug(player.roads.debug(), DARKPURPLE);
            draw_layer_debug(player.highways.debug(), DARKPURPLE);
            draw_layer_debug(player.trees.debug(), DARKPURPLE);
            draw_layer_debug(cities.debug(), DARKPURPLE);
        }

        if debug_view {
            draw_rectangle_lines(
                -padding.x,
                -padding.y,
                padding.x * 2.,
                padding.y * 2.,
                debug_zoom,
                PURPLE,
            );

            let vision_range = player.vision_range::<Roads>(padding);
            draw_bounds(vision_range, PURPLE);

            for index in player.grid_vision_range(padding).iter() {
                let current_chunk = Roads::bounds(index);
                draw_bounds(current_chunk, PURPLE);
            }
            set_camera(&overlay_camera);
            draw_text(&format!("fps: {}", get_fps()), 0., 30., 30., WHITE);
            draw_text(
                &format!(
                    "speed: {:.0}km/h",
                    player.car.body.velocity.length() * 3600. / 1000.
                ),
                0.,
                60.,
                30.,
                WHITE,
            );
            draw_multiline_text(
                &format!("{:#.2?}", player.car.body),
                0.,
                90.,
                30.,
                Some(1.),
                WHITE,
            );
        }

        next_frame().await
    }
}

fn point_to_3d(p: Point2d) -> Vec3 {
    vec3(p.x as f32, p.y as f32, 0.0)
}

const LOOK_SPEED: f32 = 10.;
const MOVE_SPEED: f32 = 1000.;

async fn render_3d_layers(top_layers: Vec<&dyn DynLayer>) {
    let levels = layer_levels(top_layers).concat();
    set_cursor_grab(true);
    show_mouse(false);
    let world_up = vec3(0.0, 0.0, 1.0);
    let mut yaw: f32 = 1.18;
    let mut pitch: f32 = 0.0;

    let mut front;
    let mut right;
    let mut up;

    let mut position = vec3(2000.0, -2000.0, 2000.);
    let mut max_level = levels.len();

    while !is_key_pressed(KeyCode::Escape) {
        let delta = get_frame_time();
        let mouse_delta = mouse_delta_position();
        yaw += mouse_delta.x * delta * LOOK_SPEED;
        pitch += mouse_delta.y * delta * LOOK_SPEED;

        pitch = if pitch > 1.5 { 1.5 } else { pitch };
        pitch = if pitch < -1.5 { -1.5 } else { pitch };

        front = vec3(
            yaw.cos() * pitch.cos(),
            yaw.sin() * pitch.cos(),
            pitch.sin(),
        )
        .normalize();

        right = front.cross(world_up).normalize();
        up = right.cross(front).normalize();

        if is_key_down(KeyCode::W) {
            position += front * delta * MOVE_SPEED;
        }
        if is_key_down(KeyCode::S) {
            position -= front * delta * MOVE_SPEED;
        }
        if is_key_down(KeyCode::D) {
            position += right * delta * MOVE_SPEED;
        }
        if is_key_down(KeyCode::A) {
            position -= right * delta * MOVE_SPEED;
        }
        if is_key_down(KeyCode::Q) {
            position += up * delta * MOVE_SPEED;
        }
        if is_key_down(KeyCode::E) {
            position -= up * delta * MOVE_SPEED;
        }
        if is_key_pressed(KeyCode::R) {
            // Always show the topmost layer
            max_level = (max_level - 1).max(1);
        }
        if is_key_pressed(KeyCode::F) {
            max_level = levels.len().min(max_level + 1);
        }

        set_camera(&Camera3D {
            position,
            up,
            target: position + front,
            ..Default::default()
        });
        clear_background(BLACK);

        for (layer_index, layer) in levels[..max_level].iter().enumerate() {
            for (bounds, chunk) in layer.iter_all_loaded() {
                let pos = vec3(0.0, 0.0, layer_index as f32 * -100.);
                let color = COLORS[layer_index % COLORS.len()];
                let max = point_to_3d(bounds.max) + pos;
                let min = point_to_3d(bounds.min) + pos;
                let mut border_color = color;
                border_color.a = 0.2;
                draw_line_3d(min, vec3(min.x, max.y, pos.z), border_color);
                draw_line_3d(min, vec3(max.x, min.y, pos.z), border_color);
                draw_line_3d(vec3(max.x, min.y, pos.z), max, border_color);
                draw_line_3d(vec3(min.x, max.y, pos.z), max, border_color);
                for thing in chunk.render() {
                    match thing {
                        DebugContent::Line(line) => draw_line_3d(
                            pos + point_to_3d(line.start),
                            pos + point_to_3d(line.end),
                            color,
                        ),
                        DebugContent::Circle { center, radius } => {
                            let center = pos + point_to_3d(center);
                            let mut x = radius;
                            let mut y = 0.;
                            for i in 1..=9 {
                                let i = (i as f32 * 10.).to_radians();
                                let (y2, x2) = i.sin_cos();
                                let x2 = x2 * radius;
                                let y2 = y2 * radius;
                                draw_line_3d(
                                    vec3(x, y, 0.) + center,
                                    vec3(x2, y2, 0.) + center,
                                    color,
                                );
                                draw_line_3d(
                                    vec3(-x, -y, 0.) + center,
                                    vec3(-x2, -y2, 0.) + center,
                                    color,
                                );
                                draw_line_3d(
                                    vec3(-x, y, 0.) + center,
                                    vec3(-x2, y2, 0.) + center,
                                    color,
                                );
                                draw_line_3d(
                                    vec3(x, -y, 0.) + center,
                                    vec3(x2, -y2, 0.) + center,
                                    color,
                                );
                                (x, y) = (x2, y2);
                            }
                        }
                        DebugContent::Text { .. } => {}
                    }
                }
            }
        }
        next_frame().await
    }
    set_cursor_grab(false);
    show_mouse(true);
}

fn layer_levels(top_layers: Vec<&dyn DynLayer>) -> Vec<Vec<&dyn DynLayer>> {
    let mut seen = BTreeMap::new();
    let mut next_layers = top_layers;
    for level in 0.. {
        for layer in std::mem::take(&mut next_layers) {
            next_layers.extend(layer.deps());
            seen.entry(layer.ident()).or_insert((level, layer)).0 = level;
        }
        if next_layers.is_empty() {
            break;
        }
    }
    let mut levels = vec![];
    for (level, layer) in seen.into_values() {
        if levels.len() < level + 1 {
            levels.resize(level + 1, vec![]);
        }
        levels[level].push(layer);
    }
    levels
}

const COLORS: [Color; 23] = [
    PURPLE, YELLOW, RED, BLUE, DARKGRAY, GOLD, PINK, DARKGREEN, LIGHTGRAY, DARKPURPLE, GREEN,
    ORANGE, BROWN, DARKBLUE, GRAY, SKYBLUE, VIOLET, BEIGE, MAROON, LIME, DARKBROWN, WHITE, MAGENTA,
];

async fn render_debug_layers(top_layers: Vec<&dyn DynLayer>) {
    let levels = layer_levels(top_layers);

    set_default_camera();
    while !is_key_pressed(KeyCode::Escape) {
        clear_background(BLACK);

        let mut positions = HashMap::new();
        let font_size = 15.;
        let mut pos = vec2(0.0, 0.0);
        let mut color = 0;
        for layers in &levels {
            pos += 10.;
            for layer in layers {
                pos.y += font_size + 10.;
                pos.x += 10.;
                let size = draw_text(&layer.name(), pos.x, pos.y, font_size, COLORS[color]);
                draw_rectangle_lines(
                    pos.x - 1.,
                    pos.y + 1.,
                    size.width + 2.,
                    -size.height - 1.,
                    1.,
                    COLORS[color],
                );
                positions.insert(layer.ident(), (pos, 3., COLORS[color]));
                color += 1;
                color %= COLORS.len();
            }
        }

        for layers in &levels {
            for layer in layers {
                let (pos, _, color) = positions[&layer.ident()];
                for dep in layer.deps() {
                    let (dep_pos, offset, _) = positions.get_mut(&dep.ident()).unwrap();
                    draw_line(pos.x, pos.y, pos.x, dep_pos.y - *offset, 1., color);
                    draw_line(
                        pos.x,
                        dep_pos.y - *offset,
                        dep_pos.x,
                        dep_pos.y - *offset,
                        1.,
                        color,
                    );
                    *offset += 3.;
                }
            }
        }
        next_frame().await
    }
}

#[derive(Debug)]
struct Car {
    // In `m`
    length: f32,
    // In `m`
    width: f32,
    body: Body,
    color: Color,
    /// Maximum angle of the front wheels, in degrees
    steering_limit: i8,
    steering: f32,
    /// Enable the braking lights
    braking: bool,
    /// Enable the reversing lights
    reversing: bool,
}

struct Actions {
    accelerate: bool,
    hand_brake: bool,
    reverse: bool,
    left: bool,
    right: bool,
}

const ENGINE_POWER: f32 = 5.;
const FRICTION: f32 = -0.0005;
const DRAG: f32 = -0.005;
const MAX_WHEEL_FRICTION_BEFORE_SLIP: f32 = 20.;

impl Car {
    // Taken from https://github.com/godotrecipes/2d_car_steering/blob/master/car.gd
    fn update(&mut self, actions: Actions) {
        let heading = Vec2::from_angle(self.body.rotation);
        // Get Inputs

        const STEERING_SPEED: f32 = 2.;
        self.steering += if actions.left {
            -STEERING_SPEED
        } else if actions.right {
            STEERING_SPEED
        } else {
            if self.steering.abs() < STEERING_SPEED {
                -self.steering
            } else {
                -self.steering.signum() * STEERING_SPEED
            }
        };
        self.steering = self
            .steering
            .clamp((-self.steering_limit).into(), self.steering_limit.into());
        let steer_dir = f32::from(self.steering).to_radians();

        self.braking = actions.hand_brake;
        self.reversing = actions.reverse;

        // Kill all movement once the car gets slow enough
        // (instead of getting closer to zero velocity in decreasingly small steps)
        if !actions.accelerate && !actions.reverse && self.body.velocity.length() < 0.05 {
            self.body.velocity = Vec2::ZERO
        }

        // Apply drag (from air on car), effectively specifying a maximum velocity.
        self.body.add_impulse(
            Vec2::ZERO,
            self.body.velocity * self.body.velocity.length() * DRAG,
        );

        let wheel_offset = heading * self.length / 2.0;

        // Calculate wheel friction forces
        let rear_impulse = self.wheel_velocity(heading, -wheel_offset, actions.hand_brake);
        let rear_impulse = slip(rear_impulse);
        self.body.add_impulse(-wheel_offset, rear_impulse);

        let front_wheel_direction = Vec2::from_angle(steer_dir).rotate(heading);
        let front_impulse = self.wheel_velocity(front_wheel_direction, wheel_offset, false);
        let front_impulse = slip(front_impulse);
        self.body.add_impulse(wheel_offset, front_impulse);

        // Accumulate car engine and brake behaviors
        if actions.reverse {
            self.body
                .add_impulse(wheel_offset, -front_wheel_direction * ENGINE_POWER);
        } else if actions.accelerate {
            let multiplier = if is_key_down(KeyCode::LeftShift) {
                10.
            } else {
                1.
            };
            self.body.add_impulse(
                wheel_offset,
                front_wheel_direction * ENGINE_POWER * multiplier,
            );
        }

        self.body.step(get_frame_time());
    }

    /// Compute and aggregate lateral and forward friction.
    /// friction is infinite up to a limit where the wheel slips (for drifting)
    fn wheel_velocity(&mut self, direction: Vec2, wheel_position: Vec2, braking: bool) -> Vec2 {
        let normal = direction.perp();
        let velocity = self.body.velocity_at_local_point(wheel_position);
        let lateral_velocity = velocity.dot(normal) * normal;
        let forward_velocity = velocity.dot(direction) * direction;
        if braking {
            -forward_velocity - lateral_velocity
        } else {
            forward_velocity * FRICTION - lateral_velocity
        }
    }

    fn draw(&self) {
        draw_rectangle_ex(
            0.,
            0.,
            self.length,
            self.width,
            DrawRectangleParams {
                offset: vec2(0.5, 0.5),
                rotation: self.body.rotation,
                color: self.color,
            },
        );
        let rotation = Vec2::from_angle(self.body.rotation) * self.length / 2.;
        draw_circle(rotation.x, rotation.y, self.width / 2., self.color);

        if self.braking || self.reversing {
            let rotation = Vec2::from_angle(self.body.rotation) * (self.length / 2. + 1.);
            draw_rectangle_ex(
                -rotation.x,
                -rotation.y,
                2.,
                self.width,
                DrawRectangleParams {
                    offset: vec2(0.5, 0.5),
                    rotation: self.body.rotation,
                    color: if self.reversing { WHITE } else { RED },
                },
            );
        }

        draw_rectangle_ex(
            rotation.x,
            rotation.y,
            2.,
            1.,
            DrawRectangleParams {
                offset: vec2(0.5, 0.5),
                rotation: self.steering.to_radians() + self.body.rotation,
                color: BLACK,
            },
        );
        draw_rectangle_ex(
            -rotation.x,
            -rotation.y,
            2.,
            1.,
            DrawRectangleParams {
                offset: vec2(0., 0.5),
                rotation: self.body.rotation,
                color: BLACK,
            },
        );
    }
}

// Reduce force if aboove the slip limit
fn slip(friction: Vec2) -> Vec2 {
    friction.clamp_length_max(MAX_WHEEL_FRICTION_BEFORE_SLIP)
}
