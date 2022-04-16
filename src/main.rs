use axum::{extract::Extension, handler::post, AddExtensionLayer, Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs::File;
use std::io::Read;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;

use tensorflow as tf;
use tf::eager::{self, raw_ops, ToTensorHandle};
use tf::Graph;
use tf::ImportGraphDefOptions;
use tf::Operation;
use tf::Session;
use tf::SessionOptions;
use tf::SessionRunArgs;
use tf::Tensor;

struct DnnModel {
    session: Session,
    ctx: eager::Context,
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

    // Create eager execution context
    let opts = eager::ContextOptions::new();
    let ctx = eager::Context::new(opts).unwrap();

    // get in/out operations
    let op_x = &graph.operation_by_name_required("input").unwrap();
    let op_output = &graph.operation_by_name_required("Identity").unwrap();

    let state = Arc::new(Mutex::new(DnnModel {
        session,
        ctx,
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
    let ctx = &model.ctx;
    let op_x = &model.op_x;
    let op_output = &model.op_output;

    // Create input tensor
    let x = payload.img.to_handle(&ctx).unwrap();
    let buf = raw_ops::decode_base64(&ctx, &x).unwrap();
    let decode_png = raw_ops::DecodePng::new().channels(3);
    let img = decode_png.call(&ctx, &buf).unwrap();
    let height = img.dim(0).unwrap();
    let width = img.dim(1).unwrap();
    let cast2float = raw_ops::Cast::new().DstT(tf::DataType::Float);

    let img = cast2float.call(&ctx, &img).unwrap();

    // [0, 1]に正規化する。255.0とすると、型の不一致でエラーになる。
    let img = raw_ops::div(&ctx, &img, &255.0f32).unwrap();

    // HWC -> NHWC に変換する
    let batch = raw_ops::expand_dims(&ctx, &img, &0).unwrap();
    // [224, 224, 3]にリサイズする。
    // ここではantialiasを有効にするために、v2のAPIを使う。
    let resize_bilinear = raw_ops::ScaleAndTranslate::new()
        .kernel_type("triangle") // bilinearのオプションに相当
        .antialias(true);
    let scale = [224.0 / height as f32, 224.0 / width as f32];
    let resized = resize_bilinear
        .call(&ctx, &batch, &[224, 224], &scale, &[0f32, 0f32])
        .unwrap();
    let x = resized.resolve().unwrap();
    let x: Tensor<f32> = unsafe { x.into_tensor() };

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
