#![allow(dead_code)]
#![allow(unused_variables)]

use std::collections::HashSet;
use std::ops::RangeInclusive;
use std::{fs, vec};

use itertools::Itertools;

fn main() {
    // day1();
    // day2();
    // day3();
    // day4();
    // day5();
    day6();
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

// fn day1() {
//     let contents = fs::read_to_string("aoc1.txt").unwrap();
//     let mut sums: Vec<u64> = contents
//         .split("\n\n")
//         .map(|single_elf| {
//             single_elf
//                 .split("\n")
//                 .filter(|s| !s.is_empty())
//                 .map(|calories| calories.parse::<u64>().unwrap())
//                 .sum::<u64>()
//         })
//         .collect();
//     sums.sort();
//     println!("{}", sums.last().unwrap());
//     println!(
//         "{}",
//         sums.pop().unwrap() + sums.pop().unwrap() + sums.pop().unwrap()
//     );
// }
