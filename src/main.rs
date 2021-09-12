use axum::{extract::Extension, handler::post, AddExtensionLayer, Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs::File;
use std::io::Read;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;

use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Operation;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Tensor;

struct DnnModel {
    session: Session,
    op_x: Operation,
    op_output: Operation,
}

#[tokio::main]
async fn main() {
    // Load the frozen_model via crate_model.py
    let mut graph = Graph::new();
    let mut proto = Vec::new();
    File::open("model/model.pb")
        .unwrap()
        .read_to_end(&mut proto)
        .unwrap();
    graph
        .import_graph_def(&proto, &ImportGraphDefOptions::new())
        .unwrap();
    let session = Session::new(&SessionOptions::new(), &graph).unwrap();

    // get in/out operations
    let op_x = &graph.operation_by_name_required("input").unwrap();
    let op_output = &graph.operation_by_name_required("Identity").unwrap();

    let state = Arc::new(Mutex::new(DnnModel {
        session,
        op_x: op_x.clone(),
        op_output: op_output.clone(),
    }));

    let app = Router::new()
        .route("/", post(proc))
        .layer(AddExtensionLayer::new(state));

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    println!("listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

#[derive(Deserialize)]
struct RequestJson {
    img: String,
}

#[derive(Serialize)]
struct ResponseJson {
    result: Vec<String>,
}

async fn proc(
    Json(payload): Json<RequestJson>,
    Extension(state): Extension<Arc<Mutex<DnnModel>>>,
) -> Json<Value> {
    let model = state.lock().await;
    let session = &model.session;
    let op_x = &model.op_x;
    let op_output = &model.op_output;

    // Create input tensor
    let x = Tensor::from(payload.img);

    // Run the graph.
    let mut args = SessionRunArgs::new();
    args.add_feed(op_x, 0, &x);
    let token_output = args.request_fetch(op_output, 0);
    session.run(&mut args).unwrap();
    // Check the output.
    let output: Tensor<f32> = args.fetch(token_output).unwrap();

    // Calculate argmax of the output
    let (max_idx, max_val) =
        output
            .iter()
            .enumerate()
            .fold((0, output[0]), |(idx_max, val_max), (idx, val)| {
                if &val_max > val {
                    (idx_max, val_max)
                } else {
                    (idx, *val)
                }
            });

    Json(json!({ "result": [max_idx, max_val] }))
}
