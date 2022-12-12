#![allow(dead_code)]
#![allow(unused_variables)]

use std::collections::{HashMap, HashSet};
use std::ops::RangeInclusive;
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
    day12();
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
