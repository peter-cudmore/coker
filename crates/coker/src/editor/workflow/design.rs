use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct BlockView {
    name: String,
    path: String,
    block_type: String,
    top: usize, 
    left: usize,
    width: usize,
    height: usize,
    inputs: Vec<Port>,
    outputs: Vec<Port>
}
#[derive(Serialize, Deserialize)]
struct Port {
    name: String,
    path: String,
    signal_type: String,
}

#[derive(Serialize, Deserialize)]
struct Point2D{
    x: usize,
    y: usize
}

#[derive(Serialize, Deserialize)]
struct Wire {
    name: String,
    input: String,
    output: String,
    path: Vec<Point2D>,
}

#[derive(Serialize, Deserialize)]
struct Composite{
    name: String,
    path: String,
    blocks: Vec<BlockView>,
    wires: Vec<Wire>
}


