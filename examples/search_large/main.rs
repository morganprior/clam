mod data_readers_tmp;

fn main() {
    println!("Hello from search-1b!");
    let index = 0;
    data_readers_tmp::read_data::<i8>(index);
    println!("Finished reading {}", data_readers_tmp::BIG_ANN_DATA[index].folder);
}
