mod hnsw;
mod node;
mod vector;
mod matrix;
mod configuration;

use configuration::{Configuration, ConfigurationInitializer};
use hnsw::HnswIndex;
use rand::Rng;
use vector::{Distance, VectorItem, DistanceType};

fn main() {

    let conf = <Configuration<f64> as ConfigurationInitializer<f64>>::init(10);
    println!{"{}",conf.m1.m};
    println!{"{}",conf.m2.m};
    for i in 0..conf.s.len() {
        print!("{}", conf.s[i])
    }
    println!("");

    let hnsw = HnswIndex::new(10000,
                                                    10,
                                                    1.0 / 3.0,
                                                    16,
                                                    Distance {_type :  DistanceType::Euclidean,});

    // Add a larger number of items
    for i in 0..1000 {
        let mut rng = rand::thread_rng();
        let vector = (0..10).map(|_| rng.gen_range(0.0..10.0)).collect();
        let item = VectorItem { id: i, vector };
        hnsw.add(item).unwrap();
    }

    let test_queries = vec![vec![1.0, 2.0], vec![5.0, 5.0], vec![9.0, 0.0], vec![0.0, 0.0], vec![0.0,0.0]];

    for query_vec in test_queries {
        let query = VectorItem {
            id: 9999,
            vector: query_vec,
        };
        match hnsw.search(&query, 5) {
            Ok(results) => {
                println!("Search results for query {:?}:", query.vector);
                for result in results {
                    println!("Found item with id: {}", result.id);
                }
            }
            Err(e) => println!("Search error: {}", e),
        }
    }
}
