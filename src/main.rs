use axum::{extract::Extension, handler::post, AddExtensionLayer, Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;

use tensorflow::eager;
use tensorflow::Graph;
use tensorflow::Operation;
use tensorflow::SavedModelBundle;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Tensor;
use tensorflow::DEFAULT_SERVING_SIGNATURE_DEF_KEY;

struct DnnModel {
    session: Session,
    op_x: Operation,
    op_output: Operation,
}

#[tokio::main]
async fn main() {
    // Load the saved model exported by zenn_savedmodel.py.
    let mut graph = Graph::new();
    let bundle =
        SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, "model").unwrap();

    // get in/out operations
    let signature = bundle
        .meta_graph_def()
        .get_signature(DEFAULT_SERVING_SIGNATURE_DEF_KEY)
        .unwrap();
    let x_info = signature.get_input("input").unwrap();
    let op_x = &graph
        .operation_by_name_required(&x_info.name().name)
        .unwrap();
    let output_info = signature.get_output("Predictions").unwrap();
    let op_output = &graph
        .operation_by_name_required(&output_info.name().name)
        .unwrap();

    let state = Arc::new(Mutex::new(DnnModel {
        session: bundle.session,
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

    let buf = base64::decode(&payload.img).unwrap();

    // Convert the buffer to a input Tensor through eager APIs (quite experimental)
    let buf = unsafe { String::from_utf8_unchecked(buf) };
    let buf = Tensor::from(buf);
    let img = eager::decode_png(buf, 3, tensorflow::DataType::UInt8).unwrap();
    let images = eager::expand_dims(img, Tensor::from(&[0])).unwrap();
    let img = eager::resize_blinear(images, Tensor::from(&[224, 224]), false, false).unwrap();
    let x: Tensor<f32> = img.resolve().unwrap();

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
