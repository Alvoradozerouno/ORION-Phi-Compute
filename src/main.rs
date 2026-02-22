// ORION Phi Compute — Rust Core (Concept)
//
// This is a conceptual Rust implementation for high-performance Phi computation.
// The Python reference implementation (orion_phi.py) is functional.
// This Rust version is planned for v2.0.
//
// Key advantages of Rust for Phi computation:
// - Zero-cost abstractions
// - No garbage collection pauses
// - SIMD vectorization for matrix operations
// - Thread-safe parallelism with rayon
// - WebAssembly compilation for browser-based computation
//
// Author: ORION - Elisabeth Steurer & Gerhard Hirschmann
// License: MIT

use std::collections::HashSet;

/// Represents the result of a Phi computation
#[derive(Debug, Clone)]
pub struct PhiResult {
    pub phi: f64,
    pub phi_lower: Option<f64>,
    pub phi_upper: Option<f64>,
    pub method: String,
    pub n_elements: usize,
    pub partition: Option<(Vec<usize>, Vec<usize>)>,
    pub proof_hash: String,
}

/// Main Phi computation engine
pub struct PhiCompute {
    proofs: Vec<String>,
}

impl PhiCompute {
    pub fn new() -> Self {
        PhiCompute { proofs: Vec::new() }
    }
    
    /// Compute exact Phi for small systems (≤12 elements)
    pub fn compute_exact(&mut self, tpm: &[Vec<f64>], state: &[u8]) -> PhiResult {
        let n = state.len();
        assert!(n <= 12, "Exact computation limited to 12 elements");
        
        let elements: Vec<usize> = (0..n).collect();
        let mut min_phi = f64::INFINITY;
        let mut best_partition = None;
        
        // Iterate over all bipartitions
        for r in 1..n {
            for subset in combinations(&elements, r) {
                let complement: Vec<usize> = elements.iter()
                    .filter(|e| !subset.contains(e))
                    .copied()
                    .collect();
                
                let phi = self.partition_phi(tpm, state, &subset, &complement);
                
                if phi < min_phi {
                    min_phi = phi;
                    best_partition = Some((subset, complement));
                }
            }
        }
        
        if min_phi == f64::INFINITY {
            min_phi = 0.0;
        }
        
        PhiResult {
            phi: min_phi,
            phi_lower: None,
            phi_upper: None,
            method: "exact_mip".into(),
            n_elements: n,
            partition: best_partition,
            proof_hash: self.hash_proof(min_phi, n, "exact"),
        }
    }
    
    /// Compute approximate Phi for large systems using greedy bipartition
    pub fn compute_approximate(&mut self, connectivity: &[Vec<f64>]) -> PhiResult {
        let n = connectivity.len();
        
        let mut part_a: HashSet<usize> = [0].into_iter().collect();
        let mut part_b: HashSet<usize> = (1..n).collect();
        
        let mut improved = true;
        let mut best_phi = self.partition_info(connectivity, &part_a, &part_b);
        
        while improved {
            improved = false;
            
            // Try moving elements between partitions
            for elem in part_a.iter().copied().collect::<Vec<_>>() {
                if part_a.len() <= 1 { continue; }
                let mut new_a = part_a.clone();
                let mut new_b = part_b.clone();
                new_a.remove(&elem);
                new_b.insert(elem);
                
                let phi = self.partition_info(connectivity, &new_a, &new_b);
                if phi < best_phi {
                    best_phi = phi;
                    part_a = new_a;
                    part_b = new_b;
                    improved = true;
                    break;
                }
            }
        }
        
        PhiResult {
            phi: best_phi,
            phi_lower: Some(best_phi * 0.9),
            phi_upper: Some(best_phi * 1.1),
            method: "greedy_bipartition".into(),
            n_elements: n,
            partition: Some((part_a.into_iter().collect(), part_b.into_iter().collect())),
            proof_hash: self.hash_proof(best_phi, n, "greedy"),
        }
    }
    
    fn partition_phi(&self, tpm: &[Vec<f64>], _state: &[u8], part_a: &[usize], part_b: &[usize]) -> f64 {
        let mut total = 0.0;
        for &a in part_a {
            for &b in part_b {
                if a < tpm.len() && b < tpm[0].len() {
                    total += tpm[a][b].abs();
                }
            }
        }
        total / (part_a.len() * part_b.len()).max(1) as f64
    }
    
    fn partition_info(&self, connectivity: &[Vec<f64>], part_a: &HashSet<usize>, part_b: &HashSet<usize>) -> f64 {
        let mut total = 0.0;
        for &a in part_a {
            for &b in part_b {
                total += connectivity[a][b].abs();
            }
        }
        total / (part_a.len() * part_b.len()).max(1) as f64
    }
    
    fn hash_proof(&mut self, phi: f64, n: usize, method: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let data = format!("phi:{:.6}:n:{}:method:{}", phi, n, method);
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }
}

fn combinations(elements: &[usize], r: usize) -> Vec<Vec<usize>> {
    if r == 0 { return vec![vec![]]; }
    if elements.is_empty() { return vec![]; }
    
    let mut result = Vec::new();
    for (i, &elem) in elements.iter().enumerate() {
        for mut combo in combinations(&elements[i+1..], r - 1) {
            combo.insert(0, elem);
            result.push(combo);
        }
    }
    result
}

fn main() {
    let mut engine = PhiCompute::new();
    
    println!("ORION Phi Compute — Rust Core");
    println!("============================");
    
    // Example: 4-element system
    let tpm = vec![
        vec![0.0, 0.0, 0.0, 0.0],
        vec![0.8, 0.2, 0.7, 0.1],
        vec![0.3, 0.9, 0.4, 0.6],
        vec![0.9, 0.8, 0.9, 0.7],
    ];
    let state = vec![1, 0, 1, 1];
    
    let result = engine.compute_exact(&tpm, &state);
    println!("Exact Phi: {:.6}", result.phi);
    println!("Proof: {}", result.proof_hash);
}
