use crate::configuration::{Configuration, ConfigurationInitializer};
use crate::node::Node;
use crate::vector::{Distance, DistanceCalculator, VectorItem};
use ordered_float::OrderedFloat;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

pub struct HnswIndex<TC,TV> {
    nodes: Arc<Mutex<HashMap<usize, Node<TC,TV>>>>,
    _max_elements: usize,
    _dim_elements: usize,
    level_lambda: f64,
    max_level: usize,
    distance_calculator: Distance,
    configuration: Configuration<TV>
}

impl HnswIndex<usize,f64> {
    pub fn new(
        max_elements: usize,
        dim_elements: usize,
        level_lambda: f64,
        max_level: usize,
        distance_calculator: Distance,
    ) -> Self {
        HnswIndex {
            nodes: Arc::new(Mutex::new(HashMap::new())),
            _max_elements: max_elements,
            _dim_elements : dim_elements,
            level_lambda,
            max_level,
            distance_calculator,
            configuration: Configuration::<f64>::init(dim_elements)
        }
    }

    fn random_layer(&self) -> usize {
        let mut rng = rand::thread_rng();
        let mut layer = 0;
        while rng.gen::<f64>() < self.level_lambda && layer < self.max_level {
            layer += 1;
        }
        layer
    }

    // [-0.5 * sum(v_i^2), p1, .., pn]
    fn extend_vector_with_first_component (v : &mut Vec<f64>) -> Vec<f64> {
        // pi^2
        let squared : Vec<f64> = v.iter().map(|x| x*x).collect();
        // sum(p_i^2)
        let first : f64 =squared.iter().sum();
        // -0.5 * sum(p_i^2)
        let mut first = [-0.5 * first].to_vec();
        // [-0.5 * sum(v_i^2), p1, .., pn]
        first.append(v);
        first
    }

    fn build_p1_p2_vectors (s : &Vec<i8>, p : &Vec<f64>, p1 : &mut Vec<f64>, p2 : &mut Vec<f64>) -> (Vec<f64>,Vec<f64>){
        let mut rng = rand::thread_rng();
        for i in 0..(s.len()) {
            if s[i] == 1 {
                p1.push(p[i] as f64);
                p2.push(p[i] as f64);
            } else {
                let rnd = rng.gen::<f64>();
                p1.push(rnd);
                p2.push(p[i] - rnd)
            }
        }
        return (p1.to_vec(),p2.to_vec())
    }

    fn encrypt_vector (&self, p : &mut Vec<f64>) -> Vec<f64> {
        // p = [-0.5 * sum(v_i^2), p1, .., pn]
        let p = Self::extend_vector_with_first_component(p);
        let (p1,p2) =
            Self::build_p1_p2_vectors(&self.configuration.s,
                                    &p,
                                 &mut Vec::<f64>::new(),
                                 &mut Vec::<f64>::new());

        let p1p =
            self.configuration.m1.m.transpose() *
            nalgebra::DMatrix::<f64>::from_vec(p1.len(),1, p1);
        let p2p =
            self.configuration.m2.m.transpose() *
            nalgebra::DMatrix::<f64>::from_vec(p2.len(),1, p2);

        let mut pp = p1p.as_slice().to_vec();
        pp.append(&mut p2p.as_slice().to_vec());
        for i in 0..pp.len() {
            println!("{}", pp[i])
        }
        println!("--end pp--");
        return pp
    }

    pub fn add(&self, mut item: VectorItem<f64>) -> Result<(), String> {
        let mut nodes = self.nodes.lock().unwrap();
        let node_id = item.id;
        let layer = self.random_layer();
        let vec = Self::encrypt_vector(&self, &mut item.vector);
        let new_node = Node {
            id: node_id,
            connections: vec![Vec::new(); self.max_level + 1],
            item: {item.vector = vec; item.clone()},
            layer,
        };
        nodes.insert(node_id, new_node);
        // Logic for connecting the node in the graph goes here
        Ok(())
    }

    pub fn search(&self, query: &VectorItem<f64>, k: usize) -> Result<Vec<VectorItem<f64>>, String> {
        let nodes = self.nodes.lock().unwrap();
        let mut top_k_items: Vec<(OrderedFloat<f64>, VectorItem<f64>)> = Vec::new();

        for node in nodes.values() {
            let dist = OrderedFloat::<f64>(self.distance_calculator.calculate(query, &node.item));
            top_k_items.push((-dist, node.item.clone()));
            top_k_items.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            if top_k_items.len() > k {
                top_k_items.pop();
            }
        }

        let mut result = Vec::new();
        for (_, item) in top_k_items.into_iter() {
            result.push(item);
        }
        Ok(result)
    }

    pub fn _search_greedy(&self, query: &VectorItem<f64>, k: usize) -> Result<Vec<VectorItem<f64>>, String> {
        let nodes = self.nodes.lock().unwrap();
        let entry_point = nodes
            .keys()
            .cloned()
            .next()
            .ok_or("No nodes in the graph")?;
        let mut candidates: HashMap<usize, f64> = HashMap::new();
        let mut visited: HashSet<usize> = HashSet::new();
        let mut closest_distance: f64;

        for layer in (0..=self.max_level).rev() {
            let mut current_node = entry_point;
            loop {
                let mut closest_node = None;
                closest_distance = f64::MAX;
                for &neighbor_id in &nodes[&current_node].connections[layer] {
                    if visited.insert(neighbor_id) {
                        let neighbor = &nodes[&neighbor_id].item;
                        let distance = self.distance_calculator.calculate(query, neighbor);
                        if distance < closest_distance {
                            closest_node = Some(neighbor_id);
                            closest_distance = distance;
                        }
                    }
                }

                if let Some(closest) = closest_node {
                    current_node = closest;
                } else {
                    break;
                }
            }

            candidates.insert(current_node, closest_distance);
        }

        let top_k_items = candidates
            .iter()
            .take(k)
            .map(|(&id, &_dist)| VectorItem {
                id,
                vector: nodes.get(&id).unwrap().item.vector.clone(),
            })
            .collect::<Vec<VectorItem<f64>>>();

        Ok(top_k_items)
    }
}
