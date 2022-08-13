// mod data_readers_tmp;
mod readers;

fn main() {
    println!("Hello from search_large!");

    let (in_dir, out_dir) = {
        let mut data_dir = std::env::current_dir().unwrap();
        data_dir.pop();
        data_dir.push("data");

        let in_dir = data_dir.join("search_large");
        assert!(in_dir.exists(), "Data Dir not found: {:?}", &in_dir);

        let out_dir = data_dir.join("working");
        readers::make_dir(&out_dir).unwrap();
        (in_dir, out_dir)
    };

    let msft_spacev = readers::BigAnnPaths {
        folder: "msft_spacev",
        train: "msft_spacev-1b.i8bin",
        subset_1: None,
        subset_2: None,
        subset_3: None,
        query: "public_query_gt100.bin",
        ground: "msspacev-1B",
    };

    readers::transform::<i8>(
        &msft_spacev,
        &in_dir.join(msft_spacev.folder),
        &out_dir.join(msft_spacev.folder),
    )
    .map_err(|reason| format!("Failed on msft_spacev because {}", reason))
    .unwrap();
}
