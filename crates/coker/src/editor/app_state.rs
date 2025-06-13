use crate::errors::CokerError;
use crate::project::Project;

pub(crate) struct AppState{
    pub project: Project,

}
impl AppState {
pub  fn new(project: Project) -> Result<Self, CokerError> {
    
        Ok(AppState{project})
    }
}
