#![allow(dead_code)]
#![allow(unused_variables)]

use std::collections::{HashMap, HashSet};
use std::ops::{RangeInclusive, Shl, Shr};
use std::{fs, vec};

use itertools::Itertools;

fn main() {
    // day1();
    // day2();
    // day3();
    // day4();
    // day5();
    // day6();
    // day7();
    // day8();
    // day9();
    // day10();
    // day11();
    // day12();
    // day13();
    // day14();
    // day15();
    // day16();
    // day17();
    // day18();
    // day19();
    // day20();
    day21();
}

fn day21() {
    let contents = fs::read_to_string("aoc21.txt").unwrap();
    let yells: HashMap<String, Yell> = contents
        .lines()
        .map(|l| l.split_once(": ").unwrap())
        .map(|(s, yell_str)| (s.to_owned(), parse(yell_str)))
        .collect();

    let result = yells["root"].calc(&yells);

    println!("{}", result);

    let result2 = yells["root"].rebalance(&yells);

    println!("{}", result2);
}

fn parse(s: &str) -> Yell {
    if let Ok(value) = s.parse::<i64>() {
        return Yell::Value(value);
    }
    let (left, rest) = s.split_once(" ").unwrap();
    let (op, right) = rest.split_once(" ").unwrap();

    Yell::Operation(left.to_string(), op.to_string(), right.to_string())
}

#[derive(Clone)]
enum Yell {
    Value(i64),
    Operation(String, String, String),
}

impl Yell {
    fn calc(&self, yells: &HashMap<String, Yell>) -> i64 {
        match self {
            Yell::Value(v) => *v,
            Yell::Operation(left, op, right) => match op.as_str() {
                "+" => yells[left].calc(yells) + yells[right].calc(yells),
                "/" => yells[left].calc(yells) / yells[right].calc(yells),
                "*" => yells[left].calc(yells) * yells[right].calc(yells),
                "-" => yells[left].calc(yells) - yells[right].calc(yells),
                _ => unreachable!(),
            },
        }
    }

    fn rebalance(&self, yells: &HashMap<String, Yell>) -> i64 {
        let Yell::Operation(left, _, right) = self else {
            unreachable!();
        };
        let mut new_yells = yells.clone();
        let mut humn_side;
        let mut other_side;

        if yells[left].find_humn(yells) {
            humn_side = left.clone();
            other_side = right.clone();
        } else {
            humn_side = right.clone();
            other_side = left.clone();
        }

        while humn_side != "humn" {
            match &yells[&humn_side] {
                Yell::Value(_) => unreachable!(),
                Yell::Operation(left, op, right) => {
                    let new_name = humn_side.clone() + "inv";
                    let (new_other_yell, new_human_side) = match op.as_str() {
                        "/" => {
                            if left == "humn" || yells[left].find_humn(yells) {
                                (
                                    Yell::Operation(right.clone(), "*".to_owned(), other_side),
                                    left,
                                )
                            } else {
                                (
                                    Yell::Operation(left.clone(), "/".to_owned(), other_side),
                                    right,
                                )
                            }
                        }
                        "+" => {
                            if left == "humn" || yells[left].find_humn(yells) {
                                (
                                    Yell::Operation(other_side, "-".to_owned(), right.clone()),
                                    left,
                                )
                            } else {
                                (
                                    Yell::Operation(other_side, "-".to_owned(), left.clone()),
                                    right,
                                )
                            }
                        }
                        "*" => {
                            if left == "humn" || yells[left].find_humn(yells) {
                                (
                                    Yell::Operation(other_side, "/".to_owned(), right.clone()),
                                    left,
                                )
                            } else {
                                (
                                    Yell::Operation(other_side, "/".to_owned(), left.clone()),
                                    right,
                                )
                            }
                        }
                        "-" => {
                            if left == "humn" || yells[left].find_humn(yells) {
                                (
                                    Yell::Operation(right.clone(), "+".to_owned(), other_side),
                                    left,
                                )
                            } else {
                                (
                                    Yell::Operation(left.clone(), "-".to_owned(), other_side),
                                    right,
                                )
                            }
                        }
                        _ => unreachable!(),
                    };
                    other_side = new_name.clone();
                    new_yells.insert(new_name, new_other_yell);
                    humn_side = new_human_side.clone();
                }
            }
        }

        new_yells[&other_side].calc(&new_yells)
    }

    fn find_humn(&self, yells: &HashMap<String, Yell>) -> bool {
        match self {
            Yell::Value(_) => false,
            Yell::Operation(left, _, right) => {
                left == "humn"
                    || right == "humn"
                    || yells[left].find_humn(yells)
                    || yells[right].find_humn(yells)
            }
        }
    }
}

fn day20() {
    let contents = fs::read_to_string("aoc20.txt").unwrap();
    let data: Vec<i64> = contents.lines().map(|l| l.parse().unwrap()).collect();
    let mut enumerated: Vec<(usize, i64)> = data
        .iter()
        .enumerate()
        .map(|(i, delta)| (i, *delta))
        .collect();
    let mut next_move = 0usize;
    let mut index = 0usize;
    while next_move < data.len() {
        if enumerated[index].0 != next_move {
            index += 1;
            continue;
        }
        let item = enumerated.remove(index);
        let mut target = (index as i64 + item.1).rem_euclid(enumerated.len() as i64) as usize;
        if target == 0 {
            target = enumerated.len();
        }
        enumerated.insert(target, item);
        next_move += 1;
    }
    let zero_index = enumerated
        .iter()
        .find_position(|(_, delta)| *delta == 0)
        .unwrap()
        .0;
    let result = enumerated[(zero_index + 1000) % enumerated.len()].1
        + enumerated[(zero_index + 2000) % enumerated.len()].1
        + enumerated[(zero_index + 3000) % enumerated.len()].1;

    println!("{}", result);

    let mut enumerated2: Vec<(usize, i64)> = data
        .iter()
        .enumerate()
        .map(|(i, delta)| (i, *delta * 811589153))
        .collect();

    for _ in 0..10 {
        for next_move in 0..enumerated2.len() {
            let index = enumerated2
                .iter()
                .find_position(|(i, _)| *i == next_move)
                .unwrap()
                .0;

            let item = enumerated2.remove(index);
            let mut target = (index as i64 + item.1).rem_euclid(enumerated2.len() as i64) as usize;
            if target == 0 {
                target = enumerated2.len();
            }
            enumerated2.insert(target, item);
        }
    }

    let zero_index = enumerated2
        .iter()
        .find_position(|(_, delta)| *delta == 0)
        .unwrap()
        .0;

    let result = enumerated2[(zero_index + 1000) % enumerated2.len()].1
        + enumerated2[(zero_index + 2000) % enumerated2.len()].1
        + enumerated2[(zero_index + 3000) % enumerated2.len()].1;

    println!("{}", result);
}

fn day19() {
    let contents = fs::read_to_string("aoc19.txt").unwrap();
    let raw_blueprints: Vec<Vec<(&str, Vec<(&str, u64)>)>> = contents
        .lines()
        .map(|l| {
            l.split_once(": ")
                .unwrap()
                .1
                .split(".")
                .filter(|s| !s.is_empty())
                .map(|robot_str| robot_str.trim().split_once(" robot costs ").unwrap())
                .map(|(robot_type_str, costs_str)| {
                    (
                        robot_type_str.strip_prefix("Each ").unwrap(),
                        costs_str
                            .split(" and ")
                            .map(|cost_str| cost_str.split_once(" ").unwrap())
                            .map(|(amount_str, cost_type)| (cost_type, amount_str.parse().unwrap()))
                            .collect(),
                    )
                })
                .collect()
        })
        .collect();

    let resource_to_index: HashMap<&str, usize> = raw_blueprints[0]
        .iter()
        .enumerate()
        .map(|(i, &(resource, _))| (resource, i))
        .collect();
    let blueprints: Vec<_> = raw_blueprints
        .iter()
        .map(|blueprint| {
            let mut robots = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
            for (robot_resource, costs) in blueprint.iter() {
                for &(resource, amount) in costs.iter() {
                    robots[resource_to_index[robot_resource]][resource_to_index[resource]] = amount;
                }
            }
            robots
        })
        .collect();

    let quality_level_sum: u64 = blueprints
        .iter()
        .enumerate()
        .map(|(i, blueprint)| blueprint_geodes(blueprint, &resource_to_index, 24) * (i as u64 + 1))
        .sum();

    println!("{}", quality_level_sum);

    let product: u64 = blueprints
        .iter()
        .take(3)
        .map(|blueprint| blueprint_geodes(blueprint, &resource_to_index, 32))
        .product();

    println!("{}", product);
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
struct MineState {
    robots: [u64; 4],
    resources: [u64; 4],
}

fn blueprint_geodes(
    blueprint: &[[u64; 4]; 4],
    resource_to_index: &HashMap<&str, usize>,
    minutes: u64,
) -> u64 {
    let mut starting_robots = [0; 4];
    starting_robots[resource_to_index["ore"]] = 1;
    let starting_resources = [0; 4];
    let starting_state = MineState {
        robots: starting_robots,
        resources: starting_resources,
    };
    let mut states: HashSet<MineState> = HashSet::from([starting_state]);
    for i in 0..(minutes - 1) {
        states = states
            .iter()
            .flat_map(|state| step(blueprint, state))
            .collect();

        if states.len() < 50000 {
            prune_states(&mut states);
        }
    }

    states
        .iter()
        .map(|state| {
            state.resources[resource_to_index["geode"]] + state.robots[resource_to_index["geode"]]
        })
        .max()
        .unwrap()
}

fn prune_states(mine_states: &mut HashSet<MineState>) {
    let mut sorted_states = Vec::from_iter(mine_states.iter());
    sorted_states.sort_by(|a, b| {
        (a.robots.iter().sum::<u64>() + a.resources.iter().sum::<u64>())
            .cmp(&(b.robots.iter().sum::<u64>() + b.resources.iter().sum::<u64>()))
    });
    *mine_states = mine_states
        .iter()
        .filter(|&state| {
            !sorted_states.iter().any(|&other| {
                other != state
                    && other
                        .resources
                        .iter()
                        .zip(state.resources.iter())
                        .all(|(o, s)| o >= s)
                    && other
                        .robots
                        .iter()
                        .zip(state.robots.iter())
                        .all(|(o, s)| o >= s)
            })
        })
        .cloned()
        .collect();
}

fn step(blueprint: &[[u64; 4]; 4], mine_state: &MineState) -> Vec<MineState> {
    let mut possible_purchases = [0; 4];
    for (i, robot_costs) in blueprint.iter().enumerate() {
        possible_purchases[i] = robot_costs
            .iter()
            .zip(mine_state.resources)
            .map(|(&cost, available)| {
                if cost == 0 {
                    u64::MAX
                } else {
                    available / cost
                }
            })
            .min()
            .unwrap()
    }

    let mut new_states = vec![];

    for i in 0..4 {
        if can_subtract_resources(&mine_state.resources, &blueprint[i]) {
            let mut new_resources = mine_state.resources.clone();
            let mut new_robots = mine_state.robots.clone();
            subtract_resources(&mut new_resources, &blueprint[i]);
            add_resources(&mut new_resources, &mine_state.robots);
            new_robots[i] += 1;
            new_states.push(MineState {
                robots: new_robots,
                resources: new_resources,
            });
        }
    }

    let mut new_resources = mine_state.resources.clone();
    add_resources(&mut new_resources, &mine_state.robots);
    new_states.push(MineState {
        robots: mine_state.robots,
        resources: new_resources,
    });

    new_states
}

fn can_subtract_resources(left: &[u64; 4], right: &[u64; 4]) -> bool {
    for i in 0..left.len() {
        if right[i] > left[i] {
            return false;
        }
    }

    true
}

fn subtract_resources(left: &mut [u64; 4], right: &[u64; 4]) {
    for i in 0..left.len() {
        left[i] -= right[i];
    }
}

fn add_resources(left: &mut [u64; 4], right: &[u64; 4]) {
    for i in 0..left.len() {
        left[i] += right[i];
    }
}

fn day18() {
    let contents = fs::read_to_string("aoc18.txt").unwrap();
    let coords: HashSet<(i64, i64, i64)> = contents
        .lines()
        .map(|s| s.split_once(",").unwrap())
        .map(|(x_str, yz_str)| (x_str.parse().unwrap(), yz_str.split_once(",").unwrap()))
        .map(|(x, (y_str, z_str))| (x, y_str.parse().unwrap(), z_str.parse().unwrap()))
        .collect();

    let total_surface = measure_surface(&mut coords.iter());
    println!("{}", total_surface);

    let mut min_x = i64::MAX;
    let mut min_y = i64::MAX;
    let mut min_z = i64::MAX;
    let mut max_x = i64::MIN;
    let mut max_y = i64::MIN;
    let mut max_z = i64::MIN;

    for &(x, y, z) in coords.iter() {
        min_x = min_x.min(x - 1);
        min_y = min_y.min(y - 1);
        min_z = min_z.min(z - 1);
        max_x = max_x.max(x + 1);
        max_y = max_y.max(y + 1);
        max_z = max_z.max(z + 1);
    }

    let mut envelope: HashSet<(i64, i64, i64)> = HashSet::new();
    envelope.extend(
        (min_x..=max_x)
            .cartesian_product(min_y..=max_y)
            .flat_map(|(x, y)| [(x, y, min_z), (x, y, max_z)]),
    );
    envelope.extend(
        (min_y..=max_y)
            .cartesian_product(min_z..=max_z)
            .flat_map(|(y, z)| [(min_x, y, z), (max_x, y, z)]),
    );
    envelope.extend(
        (min_x..=max_x)
            .cartesian_product(min_z..=max_z)
            .flat_map(|(x, z)| [(x, min_y, z), (x, max_y, z)]),
    );

    loop {
        let new_envelope: HashSet<(i64, i64, i64)> = envelope
            .iter()
            .flat_map(|&(x, y, z)| {
                [
                    (x, y, z),
                    (x + 1, y, z),
                    (x - 1, y, z),
                    (x, y + 1, z),
                    (x, y - 1, z),
                    (x, y, z + 1),
                    (x, y, z - 1),
                ]
            })
            .filter(|(x, y, z)| {
                (min_x..=max_x).contains(x)
                    && (min_y..=max_y).contains(y)
                    && (min_z..=max_z).contains(z)
                    && !coords.contains(&(*x, *y, *z))
            })
            .collect();
        if new_envelope == envelope {
            break;
        }
        envelope = new_envelope;
    }

    let envelope_surface = (max_x - min_x + 1) * (max_y - min_y + 1) * 2
        + (max_y - min_y + 1) * (max_z - min_z + 1) * 2
        + (max_z - min_z + 1) * (max_x - min_x + 1) * 2;
    let inner_surface =
        measure_surface(&mut envelope.iter().chain(coords.iter())) - envelope_surface as usize;
    let outer_surface = total_surface - inner_surface;

    println!("{}", outer_surface);
}

fn measure_surface<'a>(coords: &mut impl Iterator<Item = &'a (i64, i64, i64)>) -> usize {
    let mut exposed_midpoints_doubled: HashSet<(i64, i64, i64)> = HashSet::new();
    for (x, y, z) in coords {
        let sides = [
            (2 * x + 1, 2 * y, 2 * z),
            (2 * x - 1, 2 * y, 2 * z),
            (2 * x, 2 * y + 1, 2 * z),
            (2 * x, 2 * y - 1, 2 * z),
            (2 * x, 2 * y, 2 * z + 1),
            (2 * x, 2 * y, 2 * z - 1),
        ];
        for side in sides {
            if exposed_midpoints_doubled.contains(&side) {
                exposed_midpoints_doubled.remove(&side);
            } else {
                exposed_midpoints_doubled.insert(side);
            }
        }
    }

    exposed_midpoints_doubled.len()
}

fn day17() {
    let contents = fs::read_to_string("aoc17.txt").unwrap();
    let is_right: Vec<bool> = contents.trim().chars().map(|c| c == '>').collect();
    let floor: u16 = 0b1111111111111111;
    let walls: u16 = 0b1000000011111111;
    let shapes: Vec<Vec<u16>> = vec![
        vec![0b0001111000000000],
        vec![0b0000100000000000, 0b0001110000000000, 0b0000100000000000],
        vec![0b0000010000000000, 0b0000010000000000, 0b0001110000000000],
        vec![
            0b0001000000000000,
            0b0001000000000000,
            0b0001000000000000,
            0b0001000000000000,
        ],
        vec![0b0001100000000000, 0b0001100000000000],
    ];

    let mut finished_rows: Vec<u16> = vec![floor];
    let mut shape_index = 0usize;
    let mut move_index = 0usize;
    let mut repeat: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    let mut row_period = 0;
    let mut period = 0;
    for i in 0..10000 {
        if i == 2022 {
            println!("{}", finished_rows.len() - 1);
        }
        if repeat.contains_key(&(shape_index % shapes.len(), move_index % is_right.len())) {
            row_period = finished_rows.len()
                - repeat[&(shape_index % shapes.len(), move_index % is_right.len())].0;
            period = i - repeat[&(shape_index % shapes.len(), move_index % is_right.len())].1;
        }
        repeat.insert(
            (shape_index % shapes.len(), move_index % is_right.len()),
            (finished_rows.len(), i),
        );
        let mut shift = 0;
        let mut height = finished_rows.len() + 3;
        let shape = &shapes[shape_index % shapes.len()];
        shape_index += 1;
        loop {
            let is_right_move = is_right[move_index % is_right.len()];
            move_index += 1;
            let shift_delta = if is_right_move { 1 } else { -1 };
            if !check_collision(shape, &finished_rows, shift + shift_delta, height) {
                shift += shift_delta;
            }

            if !check_collision(shape, &finished_rows, shift, height - 1) {
                height -= 1;
            } else {
                break;
            }
        }
        let new_finished_rows_len = shape.len() + height;
        if new_finished_rows_len > finished_rows.len() {
            finished_rows.resize(new_finished_rows_len, walls);
        }
        for (i, shape_row) in shape.iter().rev().enumerate() {
            finished_rows[height + i] |= shift_row(*shape_row, shift);
        }
    }

    let possible_heights: Vec<usize> = repeat
        .iter()
        .filter(|((_, _), (height, index))| index % period == 1000000000000 % period)
        .map(|((_, _), (height, index))| height + (1000000000000 - index) / period * row_period - 1)
        .collect();
    println!("{:?}", possible_heights);
}

fn check_collision(shape: &Vec<u16>, finished_rows: &Vec<u16>, shift: i32, height: usize) -> bool {
    let walls: u16 = 0b1000000011111111;
    for (i, shape_row) in shape.iter().rev().enumerate() {
        let row = *finished_rows.get(height + i).unwrap_or(&walls);
        if shift_row(*shape_row, shift) & row != 0 {
            return true;
        }
    }
    false
}

fn shift_row(row: u16, shift: i32) -> u16 {
    if shift == 0 {
        row
    } else if shift < 0 {
        row.shl(-shift)
    } else {
        row.shr(shift)
    }
}

fn day16() {
    let contents = fs::read_to_string("aoc16.txt").unwrap();
    let valves: HashMap<&str, (u64, Vec<&str>)> = contents
        .lines()
        .map(|l| l.split_once("to valve").unwrap())
        .map(|(start, targets_str)| {
            (
                start
                    .split_once(";")
                    .unwrap()
                    .0
                    .split_once(" has flow rate=")
                    .unwrap(),
                targets_str.split_once(" ").unwrap().1.split(", ").collect(),
            )
        })
        .map(|((start, flow_rate_str), targets)| {
            (
                start.split_once(" ").unwrap().1,
                (flow_rate_str.parse().unwrap(), targets),
            )
        })
        .collect();

    let idx_to_valve: Vec<&str> = valves.keys().cloned().collect();
    let valve_to_idx: HashMap<&str, usize> = idx_to_valve
        .iter()
        .enumerate()
        .map(|(index, &valve)| (valve, index))
        .collect();
    let valves_vec: Vec<(u64, Vec<usize>)> = idx_to_valve
        .iter()
        .map(|valve| &valves[valve])
        .map(|(flow, targets_vec)| {
            (
                *flow,
                targets_vec
                    .iter()
                    .map(|&target_valve| valve_to_idx[target_valve])
                    .collect(),
            )
        })
        .collect();

    let initial_state = State {
        open_valves: 0,
        position: valve_to_idx["AA"],
        prev_position: None,
    };
    let mut best_states: HashMap<State, u64> = HashMap::new();
    best_states.insert(initial_state, 0);

    for i in 0..30 {
        let mut new_best_states: HashMap<State, u64> = HashMap::new();
        for (state, flow) in best_states {
            let released_flow: u64 = valves_vec
                .iter()
                .enumerate()
                .filter(|(index, _)| state.open_valves & (1 << index) != 0)
                .map(|(_, (flow, _))| flow)
                .sum();
            if state.open_valves & (1 << state.position) == 0 && valves_vec[state.position].0 > 0 {
                let new_state = State {
                    open_valves: state.open_valves | (1 << state.position),
                    position: state.position,
                    prev_position: None,
                };
                let current_value = new_best_states.entry(new_state).or_default();
                *current_value = (*current_value).max(flow + released_flow);
            }
            for &target_valve in valves_vec[state.position].1.iter() {
                if Some(target_valve) == state.prev_position {
                    continue;
                }
                let new_state = State {
                    open_valves: state.open_valves,
                    position: target_valve,
                    prev_position: Some(state.position),
                };
                let current_value = new_best_states.entry(new_state).or_default();
                *current_value = (*current_value).max(flow + released_flow);
            }
        }
        best_states = new_best_states;
    }
    let best_flow = best_states.values().max().unwrap();

    println!("{}", best_flow);

    let initial_state = State2 {
        open_valves: 0,
        position: valve_to_idx["AA"],
        prev_position: None,
        elephant_position: valve_to_idx["AA"],
        prev_elephant_position: None,
    };
    let mut best_states: HashMap<State2, u64> = HashMap::new();
    best_states.insert(initial_state, 0);

    for i in 0..26 {
        let mut new_best_states: HashMap<State2, u64> = HashMap::new();
        for (state, flow) in best_states {
            let mut states_me: Vec<State2> = vec![];
            let mut states_elephant: Vec<State2> = vec![];
            let released_flow: u64 = valves_vec
                .iter()
                .enumerate()
                .filter(|(index, _)| state.open_valves & (1 << index) != 0)
                .map(|(_, (flow, _))| flow)
                .sum();
            if state.open_valves & (1 << state.position) == 0 && valves_vec[state.position].0 > 0 {
                states_me.push(State2 {
                    open_valves: state.open_valves | (1 << state.position),
                    position: state.position,
                    prev_position: None,
                    elephant_position: state.elephant_position,
                    prev_elephant_position: state.prev_elephant_position,
                });
            }
            for &target_valve in valves_vec[state.position].1.iter() {
                if Some(target_valve) == state.prev_position {
                    continue;
                }
                states_me.push(State2 {
                    open_valves: state.open_valves,
                    position: target_valve,
                    prev_position: Some(state.position),
                    elephant_position: state.elephant_position,
                    prev_elephant_position: state.prev_elephant_position,
                });
            }
            if state.open_valves & (1 << state.elephant_position) == 0
                && valves_vec[state.elephant_position].0 > 0
            {
                states_elephant.push(State2 {
                    open_valves: state.open_valves | (1 << state.elephant_position),
                    position: state.position,
                    prev_position: state.prev_position,
                    elephant_position: state.elephant_position,
                    prev_elephant_position: None,
                });
            }
            for &target_valve in valves_vec[state.elephant_position].1.iter() {
                if Some(target_valve) == state.prev_elephant_position {
                    continue;
                }
                states_elephant.push(State2 {
                    open_valves: state.open_valves,
                    position: state.position,
                    prev_position: state.prev_position,
                    elephant_position: target_valve,
                    prev_elephant_position: Some(state.elephant_position),
                });
            }
            let new_flow = flow + released_flow;
            for (state_me, state_elephant) in
                states_me.iter().cartesian_product(states_elephant.iter())
            {
                let combined_state = State2 {
                    open_valves: state_me.open_valves | state_elephant.open_valves,
                    position: state_me.position,
                    prev_position: state_me.prev_position,
                    elephant_position: state_elephant.elephant_position,
                    prev_elephant_position: state_elephant.prev_elephant_position,
                };
                let current_value = new_best_states.entry(combined_state).or_default();
                *current_value = (*current_value).max(new_flow);
            }
        }
        best_states = new_best_states;
    }
    let best_flow = best_states.values().max().unwrap();

    println!("{}", best_flow);
}

#[derive(PartialEq, Eq, Hash, Clone)]
struct State {
    open_valves: u64,
    position: usize,
    prev_position: Option<usize>,
}

#[derive(PartialEq, Eq, Hash, Clone)]
struct State2 {
    open_valves: u64,
    position: usize,
    prev_position: Option<usize>,
    elephant_position: usize,
    prev_elephant_position: Option<usize>,
}

fn day15() {
    let contents = fs::read_to_string("aoc15.txt").unwrap();
    let reports: Vec<(i64, i64, i64, i64)> = contents
        .lines()
        .map(|l| l.split_once(": closest beacon is at ").unwrap())
        .map(|(left_str, right_str)| (parse_coord(left_str), parse_coord(right_str)))
        .map(|((sx, sy), (bx, by))| (sx, sy, bx, by))
        .collect();

    let impossible_xs: HashSet<_> = reports
        .iter()
        .flat_map(|&(sx, sy, bx, by)| {
            let dist = sx.abs_diff(bx) as i64 + sy.abs_diff(by) as i64;
            let row_dist = sy.abs_diff(2000000) as i64;
            (sx - dist + row_dist)..(sx + dist - row_dist)
        })
        .collect();

    println!("{}", impossible_xs.len());

    for y in 0..4000001 {
        let mut impossible_ranges: Vec<_> = reports
            .iter()
            .map(|&(sx, sy, bx, by)| {
                let dist = sx.abs_diff(bx) as i64 + sy.abs_diff(by) as i64;
                let row_dist = sy.abs_diff(y) as i64;
                (sx - dist + row_dist)..(sx + dist - row_dist + 1)
            })
            .filter(|r| !r.is_empty())
            .collect();

        impossible_ranges.sort_by(|a, b| a.start.cmp(&b.start));

        let mut max_end = 0;
        for range in impossible_ranges {
            if range.start > max_end {
                println!("{}", max_end * 4000000 + y);
                break;
            }
            if range.end > 4000000 {
                break;
            }
            max_end = max_end.max(range.end);
        }
    }
}

fn parse_coord(s: &str) -> (i64, i64) {
    let (remaining_str, y_str) = s.rsplit_once(", y=").unwrap();

    (
        remaining_str.rsplit_once("x=").unwrap().1.parse().unwrap(),
        y_str.parse().unwrap(),
    )
}

fn day14() {
    let contents = fs::read_to_string("aoc14.txt").unwrap();
    let wall_lines: HashSet<(i32, i32)> = contents
        .lines()
        .flat_map(|l| {
            l.split(" -> ")
                .map(|pair| pair.split_once(",").unwrap())
                .map(|(x, y)| (x.parse::<i32>().unwrap(), y.parse::<i32>().unwrap()))
                .tuple_windows()
                .flat_map(|((x1, y1), (x2, y2))| {
                    (x1.min(x2)..=x1.max(x2)).cartesian_product(y1.min(y2)..=y2.max(y1))
                })
        })
        .collect();

    let max_y = *wall_lines.iter().map(|(_, y)| y).max().unwrap();

    let mut blocked_places = wall_lines.clone();
    let mut sand_x = 500;
    let mut sand_y = 0;
    while sand_y < max_y {
        if !blocked_places.contains(&(sand_x, sand_y + 1)) {
            sand_y += 1;
        } else if !blocked_places.contains(&(sand_x - 1, sand_y + 1)) {
            sand_y += 1;
            sand_x -= 1;
        } else if !blocked_places.contains(&(sand_x + 1, sand_y + 1)) {
            sand_y += 1;
            sand_x += 1;
        } else {
            blocked_places.insert((sand_x, sand_y));
            sand_x = 500;
            sand_y = 0;
        }
    }
    let resting_sand = blocked_places.len() - wall_lines.len();
    println!("{}", resting_sand);

    blocked_places = wall_lines.clone();

    sand_x = 500;
    sand_y = 0;
    loop {
        if !blocked_places.contains(&(sand_x, sand_y + 1)) && sand_y < max_y + 1 {
            sand_y += 1;
        } else if !blocked_places.contains(&(sand_x - 1, sand_y + 1)) && sand_y < max_y + 1 {
            sand_y += 1;
            sand_x -= 1;
        } else if !blocked_places.contains(&(sand_x + 1, sand_y + 1)) && sand_y < max_y + 1 {
            sand_y += 1;
            sand_x += 1;
        } else {
            blocked_places.insert((sand_x, sand_y));
            if sand_y == 0 {
                break;
            }
            sand_x = 500;
            sand_y = 0;
        }
    }

    let resting_sand2 = blocked_places.len() - wall_lines.len();
    println!("{}", resting_sand2);
}

fn day13() {
    let contents = fs::read_to_string("aoc13.txt").unwrap();
    let pairs: Vec<_> = contents
        .split("\n\n")
        .map(|pair_string| pair_string.split_once("\n").unwrap())
        .map(|(p1_string, p2_string)| {
            (
                parse_packet_data(p1_string).0,
                parse_packet_data(p2_string).0,
            )
        })
        .collect();

    let ascending_count: usize = pairs
        .iter()
        .enumerate()
        .filter(|(i, (p1, p2))| p1.cmp(p2) == std::cmp::Ordering::Less)
        .map(|(i, (_, _))| i + 1)
        .sum();

    println!("{}", ascending_count);

    let mut all_packets: Vec<_> = pairs.iter().flat_map(|(p1, p2)| [p1, p2]).collect();
    let div1 = parse_packet_data("[[2]]").0;
    let div2 = parse_packet_data("[[6]]").0;
    all_packets.push(&div1);
    all_packets.push(&div2);

    all_packets.sort();
    let key: usize = all_packets
        .iter()
        .enumerate()
        .filter(|(i, &p)| *p == div1 || *p == div2)
        .map(|(i, _)| i + 1)
        .product();

    println!("{}", key);
}

fn parse_packet_data(s: &str) -> (PacketData, &str) {
    let mut remaining = if s.starts_with(",") {
        s.split_at(1).1
    } else {
        s
    };
    if remaining.starts_with("[") {
        remaining = remaining.split_at(1).1;
        let mut sub_data = vec![];
        while !remaining.starts_with("]") {
            let (data, new_remaining) = parse_packet_data(remaining);
            sub_data.push(data);
            remaining = new_remaining;
        }
        (PacketData::List(sub_data), remaining.split_at(1).1)
    } else {
        let next_closer = remaining.find("]").unwrap();
        let next_comma = remaining.find(",").unwrap_or(usize::MAX);
        let (value_string, new_remaining) = remaining.split_at(next_comma.min(next_closer));
        (
            PacketData::Value(value_string.parse().unwrap()),
            new_remaining,
        )
    }
}

#[derive(PartialEq, Eq, Clone)]
enum PacketData {
    List(Vec<PacketData>),
    Value(u64),
}

impl PartialOrd for PacketData {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PacketData {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (PacketData::List(l1), PacketData::List(l2)) => l1
                .iter()
                .zip_longest(l2.iter())
                .map(|pair| match pair {
                    itertools::EitherOrBoth::Both(p1, p2) => p1.cmp(p2),
                    itertools::EitherOrBoth::Left(_) => std::cmp::Ordering::Greater,
                    itertools::EitherOrBoth::Right(_) => std::cmp::Ordering::Less,
                })
                .find(|ord| ord.is_ne())
                .unwrap_or(std::cmp::Ordering::Equal),
            (PacketData::List(_), PacketData::Value(v2)) => {
                self.cmp(&PacketData::List(vec![PacketData::Value(*v2)]))
            }
            (PacketData::Value(v1), PacketData::List(_)) => {
                PacketData::List(vec![PacketData::Value(*v1)]).cmp(other)
            }
            (PacketData::Value(v1), PacketData::Value(v2)) => v1.cmp(v2),
        }
    }
}

fn day12() {
    let contents = fs::read_to_string("aoc12.txt").unwrap();
    let matrix: Vec<Vec<_>> = contents.lines().map(|l| l.chars().collect()).collect();
    bfs(&matrix, 'S', 'E', true);
    bfs(&matrix, 'E', 'a', false);
}

fn bfs(matrix: &Vec<Vec<char>>, start: char, end: char, is_up: bool) {
    let (start_y, start_x) = matrix
        .iter()
        .enumerate()
        .map(|(i, l)| (i, l.iter().find_position(|&c| *c == start)))
        .find_map(|(i, j)| j.map(|j2| (i, j2.0)))
        .unwrap();

    let mut open = HashSet::from([(start_x, start_y)]);
    let mut closed: HashSet<(usize, usize)> = HashSet::new();
    let mut steps = 0;
    'Bob: loop {
        let mut new_nodes: HashSet<(usize, usize)> = HashSet::new();
        for (x, y) in open.iter() {
            if matrix[*y][*x] == end {
                break 'Bob;
            }
            new_nodes.extend(
                expand(*x, *y, &matrix, is_up)
                    .iter()
                    .filter(|coord| !closed.contains(coord) && !open.contains(coord)),
            );
            closed.insert((*x, *y));
        }
        open = new_nodes;
        steps += 1;
    }

    println!("{}", steps);
}

fn expand(x: usize, y: usize, matrix: &Vec<Vec<char>>, is_up: bool) -> Vec<(usize, usize)> {
    let value = get_value(matrix[y][x]);
    let mut ret = vec![];
    if x > 0 && test_position(get_value(matrix[y][x - 1]), value, is_up) {
        ret.push((x - 1, y));
    }
    if x < matrix[y].len() - 1 && test_position(get_value(matrix[y][x + 1]), value, is_up) {
        ret.push((x + 1, y));
    }
    if y > 0 && test_position(get_value(matrix[y - 1][x]), value, is_up) {
        ret.push((x, y - 1));
    }
    if y < matrix.len() - 1 && test_position(get_value(matrix[y + 1][x]), value, is_up) {
        ret.push((x, y + 1));
    }

    ret
}

fn test_position(new_value: i32, value: i32, is_up: bool) -> bool {
    if is_up {
        new_value - value <= 1
    } else {
        new_value - value >= -1
    }
}

fn get_value(c: char) -> i32 {
    match c {
        'S' => 'a' as i32,
        'E' => 'z' as i32,
        _ => c as i32,
    }
}

fn day11() {
    let contents = fs::read_to_string("aoc11.txt").unwrap();
    let mut monkeys: Vec<Monkey> = vec![];
    let mut lines = contents.lines();
    while let Some(monkey) = parse_monkey(&mut lines) {
        monkeys.push(monkey);
    }
    let mut monkeys2 = monkeys.clone();

    let common_divisor: u64 = monkeys.iter().map(|m| m.divisor).product();

    for _ in 0..20 {
        monkey_round(&mut monkeys, common_divisor, true);
    }

    let product: usize = monkeys
        .iter()
        .map(|m| m.inspected)
        .sorted()
        .rev()
        .take(2)
        .product();

    println!("{}", product);

    for _ in 0..10000 {
        monkey_round(&mut monkeys2, common_divisor, false);
    }

    let product2: usize = monkeys2
        .iter()
        .map(|m| m.inspected)
        .sorted()
        .rev()
        .take(2)
        .product();

    println!("{}", product2);
}

fn parse_monkey<'a>(lines: &mut impl Iterator<Item = &'a str>) -> Option<Monkey> {
    if lines.next().is_none() {
        return None;
    }
    let items = lines
        .next()
        .unwrap()
        .rsplit_once(": ")
        .unwrap()
        .1
        .split(", ")
        .map(|n| n.parse::<u64>().unwrap())
        .collect();

    let operation = lines.next().unwrap();
    let operand = operation.rsplit_once(" ").unwrap().1.parse::<u64>().ok();
    let is_multiply = operation.contains("*");

    let divisor = lines
        .next()
        .unwrap()
        .rsplit_once(" ")
        .unwrap()
        .1
        .parse::<u64>()
        .unwrap();

    let target_true = lines
        .next()
        .unwrap()
        .rsplit_once(" ")
        .unwrap()
        .1
        .parse::<usize>()
        .unwrap();
    let target_false = lines
        .next()
        .unwrap()
        .rsplit_once(" ")
        .unwrap()
        .1
        .parse::<usize>()
        .unwrap();
    lines.next();

    Some(Monkey {
        items,
        operand,
        is_multiply,
        divisor,
        target_true,
        target_false,
        inspected: 0,
    })
}

fn monkey_round(monkeys: &mut Vec<Monkey>, common_divisor: u64, div3: bool) {
    for i in 0..monkeys.len() {
        let current_monkey = monkeys[i].clone();
        for item in current_monkey.items {
            let operand = current_monkey.operand.unwrap_or(item);
            let mut new_level = if current_monkey.is_multiply {
                item * operand
            } else {
                item + operand
            } % common_divisor;
            if div3 {
                new_level /= 3;
            }
            let target = if new_level % current_monkey.divisor == 0 {
                current_monkey.target_true
            } else {
                current_monkey.target_false
            };
            monkeys[target].items.push(new_level);
        }
        monkeys[i].inspected += monkeys[i].items.len();
        monkeys[i].items.clear();
    }
}

#[derive(Clone)]
struct Monkey {
    items: Vec<u64>,
    operand: Option<u64>,
    is_multiply: bool,
    divisor: u64,
    target_true: usize,
    target_false: usize,
    inspected: usize,
}

fn day10() {
    let contents = fs::read_to_string("aoc10.txt").unwrap();
    let mut x: i32 = 1;
    let mut cycle: i32 = 0;
    let mut signal_sum = 0;

    let mut screen: Vec<bool> = vec![];

    for line in contents.lines() {
        if line == "noop" {
            cycle += 1;
            if (cycle + 20) % 40 == 0 {
                signal_sum += x * cycle;
            }
            screen.push((x % 40 - (cycle - 1) % 40).abs() <= 1);
            continue;
        }
        let delta = line.split_once(" ").unwrap().1.parse::<i32>().unwrap();
        cycle += 1;
        if (cycle + 20) % 40 == 0 {
            signal_sum += x * cycle;
        }
        screen.push((x % 40 - (cycle - 1) % 40).abs() <= 1);
        cycle += 1;
        if (cycle + 20) % 40 == 0 {
            signal_sum += x * cycle;
        }
        screen.push((x % 40 - (cycle - 1) % 40).abs() <= 1);
        x += delta;
    }

    println!("{}", signal_sum);

    for (i, b) in screen.iter().enumerate() {
        print!("{}", if *b { '#' } else { '.' });
        if i % 40 == 39 {
            println!();
        }
    }
}

fn day9() {
    let contents = fs::read_to_string("aoc9.txt").unwrap();
    let pairs: Vec<(char, i32)> = contents
        .lines()
        .map(|l| l.split_once(" ").unwrap())
        .map(|(cs, n)| (cs.chars().next().unwrap(), n.parse().unwrap()))
        .collect();

    let mut head_x = 0;
    let mut head_y = 0;
    let mut tail_x = 0;
    let mut tail_y = 0;

    let mut visited: HashSet<(i32, i32)> = HashSet::from([(0, 0)]);

    for (dir, num_steps) in pairs.iter() {
        let (dx, dy) = match dir {
            'R' => (1, 0),
            'L' => (-1, 0),
            'U' => (0, 1),
            'D' => (0, -1),
            _ => unreachable!(),
        };
        for _ in 0..*num_steps {
            head_x += dx;
            head_y += dy;
            (tail_x, tail_y) = move_tail(head_x, head_y, tail_x, tail_y);
            visited.insert((tail_x, tail_y));
        }
    }
    println!("{}", visited.len());
    visited.clear();

    let mut positions: Vec<(i32, i32)> = vec![
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    ];
    for (dir, num_steps) in pairs.iter() {
        let (dx, dy) = match dir {
            'R' => (1, 0),
            'L' => (-1, 0),
            'U' => (0, 1),
            'D' => (0, -1),
            _ => unreachable!(),
        };
        for _ in 0..*num_steps {
            let (x, y) = positions[0];
            positions[0] = (x + dx, y + dy);
            for i in 0..9 {
                head_x += dx;
                head_y += dy;
                positions[i + 1] = move_tail(
                    positions[i].0,
                    positions[i].1,
                    positions[i + 1].0,
                    positions[i + 1].1,
                );
            }
            visited.insert(positions[9]);
        }
    }

    println!("{}", visited.len());
}

fn move_tail(head_x: i32, head_y: i32, tail_x: i32, tail_y: i32) -> (i32, i32) {
    let dx = head_x - tail_x;
    let dy = head_y - tail_y;
    if dx.abs() != 2 && dy.abs() != 2 {
        return (tail_x, tail_y);
    }

    (head_x - dx / 2, head_y - dy / 2)
}

fn day8() {
    let contents = fs::read_to_string("aoc8.txt").unwrap();
    let tree_matrix: Vec<Vec<_>> = contents.lines().map(|l| l.bytes().collect()).collect();

    let height = tree_matrix.len();
    let width = tree_matrix[0].len();

    let mut visible: Vec<Vec<bool>> = (0..height)
        .map(|_| (0..width).map(|_| false).collect())
        .collect();

    for y in 0..height {
        let mut max_seen = -1;
        for x in 0..width {
            let current_height = tree_matrix[y][x] as i32;
            if max_seen < current_height {
                max_seen = current_height;
                visible[y][x] = true;
            }
        }
        max_seen = -1;
        for x in (0..width).rev() {
            let current_height = tree_matrix[y][x] as i32;
            if max_seen < current_height {
                max_seen = current_height;
                visible[y][x] = true;
            }
        }
    }

    for x in 0..width {
        let mut max_seen = -1;
        for y in 0..height {
            let current_height = tree_matrix[y][x] as i32;
            if max_seen < current_height {
                max_seen = current_height;
                visible[y][x] = true;
            }
        }
        max_seen = -1;
        for y in (0..height).rev() {
            let current_height = tree_matrix[y][x] as i32;
            if max_seen < current_height {
                max_seen = current_height;
                visible[y][x] = true;
            }
        }
    }

    let num_seen = visible
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&visible| *visible)
        .count();

    println!("{}", num_seen);

    let max_tree = (0..height)
        .cartesian_product(0..width)
        .map(|(y, x)| scenic_score(x, y, &tree_matrix))
        .max()
        .unwrap();

    println!("{}", max_tree);
}

fn scenic_score(x: usize, y: usize, tree_matrix: &Vec<Vec<u8>>) -> u64 {
    let height = tree_matrix.len();
    let width = tree_matrix[0].len();
    if x == 0 || x == width || y == 0 || y == height {
        return 0;
    }

    let tree_height = tree_matrix[y][x];

    let mut right = 0;
    for x2 in (x + 1)..width {
        right += 1;
        if tree_matrix[y][x2] >= tree_height {
            break;
        }
    }

    let mut left = 0;
    for x2 in (0..x).rev() {
        left += 1;
        if tree_matrix[y][x2] >= tree_height {
            break;
        }
    }

    let mut down = 0;
    for y2 in (y + 1)..height {
        down += 1;
        if tree_matrix[y2][x] >= tree_height {
            break;
        }
    }

    let mut up = 0;
    for y2 in (0..y).rev() {
        up += 1;
        if tree_matrix[y2][x] >= tree_height {
            break;
        }
    }

    up * down * left * right
}

fn day7() {
    let contents = fs::read_to_string("aoc7.txt").unwrap();
    let mut dir_stack: Vec<String> = vec!["".to_string()];
    let mut files: HashMap<String, u64> = HashMap::new();
    let mut dirs: HashMap<String, HashSet<String>> = HashMap::new();

    for line in contents.lines() {
        if line.starts_with("$ cd") {
            let new_dir = line.split_once("cd ").unwrap().1;
            if new_dir == ".." {
                dir_stack.pop();
            } else {
                let s = dir_stack.last().unwrap().to_string() + "/" + new_dir;
                dir_stack.push(s);
            }
            continue;
        }
        if line.starts_with("$ ls") {
            continue;
        }
        if line.starts_with("dir") {
            let name = line.split_once(" ").unwrap().1;
            dirs.entry(dir_stack.last().unwrap().clone())
                .or_default()
                .insert(dir_stack.last().unwrap().to_string() + "/" + name);
            continue;
        }
        let (size, name) = line.split_once(" ").unwrap();
        files.insert(
            dir_stack.last().unwrap().to_string() + "/" + name,
            size.parse().unwrap(),
        );
        dirs.entry(dir_stack.last().unwrap().clone())
            .or_default()
            .insert(dir_stack.last().unwrap().to_string() + "/" + name);
    }

    let sizes = calc_sizes(&files, &dirs, &"//");

    let sum: u64 = sizes
        .iter()
        .filter(|(path, &size)| dirs.contains_key(*path) && size <= 100000)
        .map(|(_, size)| size)
        .sum();

    let needed_size = sizes["//"] - 40000000;

    let dir_to_del = sizes
        .iter()
        .filter(|(path, &size)| dirs.contains_key(*path) && size >= needed_size)
        .map(|(_, size)| size)
        .min()
        .unwrap();

    println!("{}", sum);
    println!("{}", dir_to_del);
}

fn calc_sizes(
    files: &HashMap<String, u64>,
    dirs: &HashMap<String, HashSet<String>>,
    path: &str,
) -> HashMap<String, u64> {
    let mut ret = HashMap::new();
    if let Some(contents) = dirs.get(path) {
        ret.insert(path.to_string(), 0);
        for new_path in contents {
            ret.extend(calc_sizes(files, dirs, new_path));
            let size = ret[new_path];
            *ret.entry(path.to_string()).or_default() += size;
        }
    }
    if let Some(size) = files.get(path) {
        ret.insert(path.to_string(), *size);
    }

    ret
}

fn day6() {
    let contents = fs::read_to_string("aoc6.txt").unwrap();
    let signal: Vec<_> = contents.bytes().collect();
    let first_index = signal
        .windows(4)
        .enumerate()
        .find_map(|(i, a)| {
            let hashset: HashSet<_> = a.iter().collect();
            if hashset.len() == a.len() {
                Some(i + 4)
            } else {
                None
            }
        })
        .unwrap();

    let first_index2 = signal
        .windows(14)
        .enumerate()
        .find_map(|(i, a)| {
            let hashset: HashSet<_> = a.iter().collect();
            if hashset.len() == a.len() {
                Some(i + 14)
            } else {
                None
            }
        })
        .unwrap();

    println!("{}", first_index);
    println!("{}", first_index2);
}

fn day5() {
    let contents = fs::read_to_string("aoc5.txt").unwrap();
    let (stacks_str, instructions_str) = contents.split_once("\n\n").unwrap();
    let mut stacks: Vec<Vec<char>> = stacks_str
        .lines()
        .flat_map(|l| {
            l.chars()
                .skip(1)
                .step_by(4)
                .enumerate()
                .filter(|(_, c)| *c != ' ')
        })
        .fold(vec![], |mut acc, (i, c)| {
            if acc.len() <= i {
                acc.resize(i + 1, vec![]);
            }
            acc[i].insert(0, c);

            acc
        });

    let instructions: Vec<Instruction> = instructions_str
        .lines()
        .map(|l| l.split_once(" from ").unwrap())
        .map(|(p1, p2)| {
            (
                p1.split_once(" ").unwrap().1,
                p2.split_once(" to ").unwrap(),
            )
        })
        .map(|(count_str, (from_str, to_str))| Instruction {
            count: count_str.parse().unwrap(),
            from: from_str.parse().unwrap(),
            to: to_str.parse().unwrap(),
        })
        .collect();

    let mut stacks2 = stacks.clone();
    let mut temp_stack = vec![];

    for instruction in instructions {
        for _ in 0..instruction.count {
            let c = stacks[instruction.from - 1].pop().unwrap();
            stacks[instruction.to - 1].push(c);
            temp_stack.push(c);
        }
        for _ in 0..instruction.count {
            temp_stack.push(stacks2[instruction.from - 1].pop().unwrap());
        }
        for _ in 0..instruction.count {
            stacks2[instruction.to - 1].push(temp_stack.pop().unwrap());
        }
    }

    let top = stacks.iter().filter_map(|v| v.last()).join("");
    let top2 = stacks2.iter().filter_map(|v| v.last()).join("");

    println!("{}", top);
    println!("{}", top2);
}

struct Instruction {
    count: usize,
    from: usize,
    to: usize,
}

fn day4() {
    let contents = fs::read_to_string("aoc4.txt").unwrap();
    let ranges = contents
        .lines()
        .map(|l| l.split_once(",").unwrap())
        .map(|(sr1, sr2)| (parse_range(sr1), parse_range(sr2)));

    let containing_count = ranges
        .clone()
        .filter(|(r1, r2)| {
            (r1.contains(r2.start()) && r1.contains(r2.end()))
                || (r2.contains(r1.start()) && r2.contains(r1.end()))
        })
        .count();

    let overlapping_count = ranges
        .filter(|(r1, r2)| {
            r1.contains(r2.start())
                || r1.contains(r2.end())
                || r2.contains(r1.start())
                || r2.contains(r1.end())
        })
        .count();

    println!("{}", containing_count);
    println!("{}", overlapping_count);
}

fn parse_range(s: &str) -> RangeInclusive<u64> {
    let (start, end) = s.split_once("-").unwrap();
    return start.parse().unwrap()..=end.parse().unwrap();
}

fn day3() {
    let contents = fs::read_to_string("aoc3.txt").unwrap();
    let priority_sum: u64 = contents
        .lines()
        .map(|l| l.chars().collect::<Vec<_>>())
        .map(|v| {
            (
                v.iter().take(v.len() / 2).cloned().collect::<HashSet<_>>(),
                v.iter().skip(v.len() / 2).cloned().collect::<HashSet<_>>(),
            )
        })
        .map(|(rucksack_a, rucksack_b)| *rucksack_a.intersection(&rucksack_b).next().unwrap())
        .map(|c| priority(c))
        .sum();

    let priority_sum2: u64 = contents
        .lines()
        .map(|l| l.chars().collect::<HashSet<_>>())
        .tuples()
        .map(|(s1, s2, s3)| {
            *s1.intersection(&s2)
                .cloned()
                .collect::<HashSet<_>>()
                .intersection(&s3)
                .next()
                .unwrap()
        })
        .map(|c| priority(c))
        .sum();

    println!("{}", priority_sum);
    println!("{}", priority_sum2);
}

fn priority(c: char) -> u64 {
    if ('a'..='z').contains(&c) {
        return (c as u64) - ('a' as u64) + 1;
    }
    if ('A'..='Z').contains(&c) {
        return (c as u64) - ('A' as u64) + 27;
    }
    panic!();
}

fn day2() {
    let contents = fs::read_to_string("aoc2.txt").unwrap();
    let score: u64 = contents
        .lines()
        .map(|l| l.split_once(" ").unwrap())
        .map(|(theirs_str, mine_str)| {
            let theirs_num = match theirs_str {
                "A" => 1,
                "B" => 2,
                "C" => 3,
                _ => panic!(""),
            };
            let mine_num = match mine_str {
                "X" => 1,
                "Y" => 2,
                "Z" => 3,
                _ => panic!(""),
            };
            let result_score = if (theirs_num % 3) == (mine_num - 1) {
                6
            } else if theirs_num == mine_num {
                3
            } else {
                0
            };

            mine_num + result_score
        })
        .sum();

    let score2: u64 = contents
        .lines()
        .map(|l| l.split_once(" ").unwrap())
        .map(|(theirs_str, mine_str)| {
            let theirs_num = match theirs_str {
                "A" => 1,
                "B" => 2,
                "C" => 3,
                _ => panic!(""),
            };
            let mine_num = match mine_str {
                "X" => (theirs_num + 1) % 3 + 1,
                "Y" => theirs_num + 3,
                "Z" => (theirs_num % 3) + 1 + 6,
                _ => panic!(""),
            };
            mine_num
        })
        .sum();

    println!("{}", score);
    println!("{}", score2);
}

fn day1() {
    let contents = fs::read_to_string("aoc1.txt").unwrap();
    let mut sums: Vec<u64> = contents
        .split("\n\n")
        .map(|single_elf| {
            single_elf
                .split("\n")
                .filter(|s| !s.is_empty())
                .map(|calories| calories.parse::<u64>().unwrap())
                .sum::<u64>()
        })
        .collect();
    sums.sort();
    println!("{}", sums.last().unwrap());
    println!(
        "{}",
        sums.pop().unwrap() + sums.pop().unwrap() + sums.pop().unwrap()
    );
}
