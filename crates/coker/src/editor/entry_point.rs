use std::path::{Path, PathBuf};
use crate::errors::CokerError;
use axum::{
    Router,
    Json,
    routing::{get, post },
    extract::{State}
};
use std::sync::Arc;
use super::app_state::AppState;
use crate::project::Project;

pub fn open_editor(project: Project) -> Result<(), CokerError> {
    
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(app_main(project))
}

async fn app_main(project: Project) -> Result<(), CokerError>{
    let app_state = Arc::new(AppState::new(project)?);
    let app = Router::new()
        .route("/", get(root))
        .with_state(app_state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080")
        .await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn root(State(state): State<Arc<AppState>>) -> Json<Project> {
    let s= &state;
        
    s.project.clone().into()
}