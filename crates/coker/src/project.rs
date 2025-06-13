use crate::errors::CokerError;

use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::E;
use std::path::{Path, PathBuf};
use toml;
use toml::Table;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ComponentItem {
    name: String,
    module: String,
    path: PathBuf,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum Workflow {
    Design,
    Simulation,
    Analysis,
    Optimisation,
    Deploy,
}

impl TryFrom<&String> for Workflow {
    type Error = ();
    fn try_from(value: &String) -> Result<Self, ()> {
        if value.eq("Design") {
            Ok(Workflow::Design)
        } else if value.eq("Simulation") {
            Ok(Workflow::Simulation)
        } else if value.eq("Analysis") {
            Ok(Workflow::Analysis)
        } else if value.eq("Optimisation") {
            Ok(Workflow::Optimisation)
        }else if value.eq("Deploy") {
            Ok(Workflow::Deploy)
        } else { 
            Err(())
        }
        
        
    }
}


#[derive(Serialize, Deserialize, Debug, Clone)]
struct WorksheetItem {
    name: String,
    path: PathBuf,
    workflow: Workflow,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum CokerItem {
    Component(ComponentItem),
    Worksheet(WorksheetItem),
}

#[derive(Deserialize, Serialize, Clone)]
pub struct Project {
    path: String,
    components: Vec<ComponentItem>,
    workflows: Vec<WorksheetItem>,
    libraries: Vec<String>,
}

impl Project {
    pub fn load(path: &Path) -> Result<Self, CokerError> {
        // check if
        import_python_package(path)?;
        let result = Project {
            path: path.to_str().unwrap().to_string(),
            components: list_components(path),
            workflows: list_worksheets(path),
            libraries: list_libraries(path),
        };
        Ok(result)
    }
}

struct RegistryItem {
    name: String,
    subdir: String,
}

fn import_python_package(path: &Path) -> Result<String, CokerError> {
    let pyproject = {
        let file = path.join("pyproject.toml");
        let contents = std::fs::read_to_string(file)?;
        let table = contents.parse::<Table>().map_err(|e| {
            CokerError::InvalidProject(path.into(), "Unable to parse pyproject.toml".to_string())
        })?;
        table
    };
    let name = match pyproject.get("project") {
        Some(e) => match e {
            toml::Value::Table(t) => {
                if let Some(name) = t.get("name") {
                    name.as_str().unwrap().to_string()
                } else {
                    return Err(CokerError::InvalidProject(
                        path.into(),
                        "Cannot find key: 'name'".to_string(),
                    ));
                }
            }
            _ => {
                return Err(CokerError::InvalidProject(
                    path.into(),
                    "Key 'project' should be a table".to_string(),
                ))
            }
        },
        None => {
            return Err(CokerError::InvalidProject(
                path.into(),
                "Cannot find project".to_string(),
            ))
        }
    };

    Python::with_gil(|py| -> PyResult<_> {
        let paths = py.import("sys")?.getattr("path")?;
        let syspath = paths.downcast::<PyList>().unwrap();
        syspath.insert(0, path.to_str().unwrap().to_string())?;
        py.import(name.as_str())?;
        Ok(())
    })
    .map_err(|e| CokerError::InvalidProject(path.into(), e.to_string()))?;

    Ok(name)
}

fn list_components(path: &Path) -> Vec<ComponentItem> {
    let objects = Python::with_gil(|py| -> PyResult<Vec<ComponentItem>> {
        let coker = py.import("coker")?;
        let registry: HashMap<String, String> =
            coker.call_method0("get_component_registry")?.extract()?;
        let items = registry
            .iter()
            .filter_map(|(k, v)| {
                let obj_path = Path::new(v);
                if obj_path.starts_with(path) {
                    let subdir = obj_path
                        .strip_prefix(path)
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_string();
                    let name = k.clone();
                    Some(
                        ComponentItem{
                            name, 
                            path: PathBuf::from(obj_path),
                            module: String::new(),
                        }
                    )
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        Ok(items)
    })
    .expect("Failed to get component registry");

    // components are in
    // dir
    //   - subdir
    //      -file
    //        - component
    //   - file
    //

    objects
}

fn list_libraries(path: &Path) -> Vec<String> {
    Vec::new()
}

fn list_worksheets(path: &Path) -> Vec<WorksheetItem> {
    let objects = Python::with_gil(|py| -> PyResult<Vec<WorksheetItem>> {
        let coker = py.import("coker")?;
        let registry: Vec<(String, String, String)> =
            coker.call_method0("get_worksheets")?.extract()?;
        let items = registry
            .iter()
            .filter_map(|(t,n, v)| {
                let obj_path = Path::new(v);
                let workflow = Workflow::try_from(t).ok()?;
                if obj_path.starts_with(path) {
                    let subdir = obj_path
                        .strip_prefix(path)
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_string();
                    let name = n.clone();
                    
                    
                    Some(

                            WorksheetItem{
                                name,
                                path: PathBuf::from(subdir),
                            workflow                                
                        }
                        
                    )
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        Ok(items)
    })
    .expect("Failed to get worksheets");
    println!("Worksheets: {:?}", objects);
    objects
    
}
