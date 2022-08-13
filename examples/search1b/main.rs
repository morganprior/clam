mod data_readers;

fn main() {
    println!("Hello from search-1b!");
    data_readers::read_data::<f32>(0);
    println!("Finished reading {}", data_readers::BIG_ANN_DATA[0].folder);
}
