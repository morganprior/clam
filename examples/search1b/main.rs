mod data_readers;

fn main() {
    println!("Hello from search-1b!");
    let index = 0;
    data_readers::read_data::<i8>(index);
    println!("Finished reading {}", data_readers::BIG_ANN_DATA[index].folder);
}
