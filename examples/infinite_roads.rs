use ::rand::prelude::*;
use ::tracing::{debug, trace};
use macroquad::{prelude::*, time};
use miniquad::window::screen_size;
use std::sync::Arc;

use layer_proc_gen::*;
use rolling_grid::{GridIndex, GridPoint, RollingGrid};
use vec2::{GridBounds, Line, Point2d};

#[path = "../tests/tracing.rs"]
mod tracing_helper;
use tracing_helper::*;

#[derive(Default)]
struct Locations(RollingGrid<Self>);
#[derive(PartialEq, Debug)]
struct LocationsChunk {
    points: [Point2d; 3],
}

impl Layer for Locations {
    type Chunk = LocationsChunk;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.0
    }

    fn ensure_all_deps(&self, _chunk_bounds: GridBounds) {}
}

impl Chunk for LocationsChunk {
    type Layer = Locations;

    fn compute(_layer: &Self::Layer, index: GridPoint) -> Self {
        let chunk_bounds = Self::bounds(index);
        trace!(?chunk_bounds);
        let mut x = SmallRng::seed_from_u64(index.x.0 as u64);
        let mut y = SmallRng::seed_from_u64(index.y.0 as u64);
        let mut seed = [0; 32];
        x.fill_bytes(&mut seed[..16]);
        y.fill_bytes(&mut seed[16..]);
        let mut rng = SmallRng::from_seed(seed);
        let points = [
            chunk_bounds.sample(&mut rng),
            chunk_bounds.sample(&mut rng),
            chunk_bounds.sample(&mut rng),
        ];
        debug!(?points);
        LocationsChunk { points }
    }
}

/// Removes locations that are too close to others
struct ReducedLocations {
    grid: RollingGrid<Self>,
    raw_locations: LayerDependency<Locations, 256, 256>,
}

#[derive(PartialEq, Debug)]
struct ReducedLocationsChunk {
    points: [Option<Point2d>; 3],
}

impl Layer for ReducedLocations {
    type Chunk = ReducedLocationsChunk;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.grid
    }

    fn ensure_all_deps(&self, chunk_bounds: GridBounds) {
        self.raw_locations.ensure_loaded_in_bounds(chunk_bounds);
    }
}

impl Chunk for ReducedLocationsChunk {
    type Layer = ReducedLocations;

    fn compute(layer: &Self::Layer, index: GridPoint) -> Self {
        let points = layer.raw_locations.get(index).points.map(|p| {
            for other in layer.raw_locations.get_range(GridBounds {
                min: p,
                max: p + Point2d::splat(100),
            }) {
                for other in other.points {
                    if other == p {
                        continue;
                    }
                    if other.dist_squared(p) < 100 * 100 {
                        return None;
                    }
                }
            }
            Some(p)
        });
        ReducedLocationsChunk { points }
    }
}

struct Roads {
    grid: RollingGrid<Self>,
    locations: LayerDependency<ReducedLocations, 256, 256>,
}

#[derive(PartialEq, Debug)]
struct RoadsChunk {
    roads: Vec<Line>,
}

impl Layer for Roads {
    type Chunk = RoadsChunk;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.grid
    }

    #[track_caller]
    fn ensure_all_deps(&self, chunk_bounds: GridBounds) {
        self.locations.ensure_loaded_in_bounds(chunk_bounds);
    }
}

impl Chunk for RoadsChunk {
    type Layer = Roads;

    fn compute(layer: &Self::Layer, index: GridPoint) -> Self {
        let mut roads = vec![];
        let mut points = [None; 3 * 9];
        for (i, point) in layer
            .locations
            .get_grid_range(GridBounds::point(index).pad(Point2d::splat(1).map(GridIndex)))
            .flat_map(|grid| grid.points.into_iter())
            .enumerate()
        {
            points[i] = point;
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
        for (i, &a) in points.iter().enumerate().skip(3 * 4).take(3) {
            if let Some(a) = a {
                for &b in points.iter().skip(i + 1) {
                    if let Some(b) = b {
                        let dist = a.dist_squared(b);
                        if points.iter().copied().flatten().all(|c| {
                            if a == c || b == c {
                                return true;
                            }
                            // FIXME: make cheaper by already bailing if `x*x` is larger than dist,
                            // to avoid computing `y*y`.
                            let a_dist = a.dist_squared(c);
                            let b_dist = c.dist_squared(b);
                            dist < a_dist || dist < b_dist
                        }) {
                            roads.push(a.to(b))
                        }
                    }
                }
            }
        }
        debug!(?roads);
        RoadsChunk { roads }
    }
}

struct Player {
    grid: RollingGrid<Self>,
    roads: LayerDependency<Roads, 1000, 1000>,
}

impl Player {
    pub fn new(roads: Arc<Roads>) -> Self {
        Self {
            grid: Default::default(),
            roads: roads.into(),
        }
    }
}

#[derive(PartialEq, Debug)]
struct PlayerChunk;

impl Layer for Player {
    type Chunk = PlayerChunk;

    fn rolling_grid(&self) -> &RollingGrid<Self> {
        &self.grid
    }

    const GRID_SIZE: Point2d<u8> = Point2d::splat(3);

    const GRID_OVERLAP: u8 = 2;

    fn ensure_all_deps(&self, chunk_bounds: GridBounds) {
        self.roads.ensure_loaded_in_bounds(chunk_bounds);
    }
}

impl Chunk for PlayerChunk {
    type Layer = Player;

    fn compute(_layer: &Self::Layer, _index: GridPoint) -> Self {
        PlayerChunk
    }
}

#[macroquad::main("layer proc gen demo")]
async fn main() {
    init_tracing();

    let mut camera = Camera2D::default();
    let standard_zoom = Vec2::from(screen_size()).recip() * 2.;
    camera.zoom = standard_zoom;
    set_camera(&camera);
    let mut overlay_camera = Camera2D::default();
    overlay_camera.zoom = standard_zoom;
    overlay_camera.offset = vec2(-1., 1.);

    let raw_locations = Arc::new(Locations::default());
    let locations = Arc::new(ReducedLocations {
        grid: Default::default(),
        raw_locations: raw_locations.into(),
    });
    let roads = Arc::new(Roads {
        grid: Default::default(),
        locations: locations.into(),
    });
    let player = Player::new(roads.clone());
    let mut player_pos = Vec2::new(0., 0.);
    let mut rotation = 0.0;
    let mut speed: f32 = 0.0;
    let mut last_load_time = 0.;
    let mut smooth_cam_rotation = rotation;
    let mut smooth_cam_speed = 0.0;
    loop {
        if is_key_down(KeyCode::W) {
            speed += 0.01;
        } else {
            speed *= 0.99;
        }
        if is_key_down(KeyCode::A) {
            rotation -= f32::to_radians(1.);
        }
        if is_key_down(KeyCode::S) {
            speed -= 0.1;
        }
        if is_key_down(KeyCode::D) {
            rotation += f32::to_radians(1.);
        }
        speed = speed.clamp(0.0, 2.0);
        player_pos += Vec2::from_angle(rotation) * speed;

        smooth_cam_rotation = smooth_cam_rotation * 0.99 + rotation * 0.01;
        camera.rotation = -smooth_cam_rotation.to_degrees() - 90.;
        smooth_cam_speed = smooth_cam_speed * 0.99 + speed * 0.01;
        camera.zoom = standard_zoom * (3.1 - smooth_cam_speed);
        set_camera(&camera);

        // Avoid moving everything in whole pixels and allow for smooth sub-pixel movement instead
        let adjust = -player_pos.fract();

        let player_pos = Point2d {
            x: player_pos.x as i64,
            y: player_pos.y as i64,
        };
        let load_time = time::get_time();
        player.ensure_loaded_in_bounds(GridBounds::point(player_pos));
        let load_time = ((time::get_time() - load_time) * 10000.).round() / 10.;
        if load_time > 0. {
            last_load_time = load_time;
        }

        clear_background(DARKGREEN);

        let point2screen = |point: Point2d| -> Vec2 {
            let point = point - player_pos;
            i64vec2(point.x, point.y).as_vec2() + adjust
        };

        let vision_range = GridBounds::point(player_pos).pad(player.roads.padding());
        trace!(?vision_range);
        for roads in player.roads.get_range(vision_range) {
            for &line in roads.roads.iter() {
                let start = point2screen(line.start);
                let end = point2screen(line.end);
                draw_line(start.x, start.y, end.x, end.y, 40., GRAY);
                draw_circle(start.x, start.y, 20., GRAY);
                draw_circle(start.x, start.y, 2., WHITE);
                draw_circle(end.x, end.y, 20., GRAY);
                draw_circle(end.x, end.y, 2., WHITE);
            }
        }
        for roads in player.roads.get_range(vision_range) {
            for &line in roads.roads.iter() {
                let start = point2screen(line.start);
                let end = point2screen(line.end);
                draw_line(start.x, start.y, end.x, end.y, 4., WHITE);
            }
        }
        draw_rectangle_ex(
            0.,
            0.,
            15.0,
            10.0,
            DrawRectangleParams {
                offset: vec2(0.5, 0.5),
                rotation,
                color: RED,
            },
        );
        let rotation = Vec2::from_angle(rotation) * 7.5;
        draw_circle(rotation.x, rotation.y, 5., RED);

        set_camera(&overlay_camera);
        draw_text(
            &format!("last load time: {last_load_time}ms"),
            0.,
            10.,
            10.,
            WHITE,
        );

        next_frame().await
    }
}