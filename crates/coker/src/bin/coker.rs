
use clap::{Args, Parser, Subcommand};
use std::ffi::OsString;
use std::fmt::{Debug};
use std::path::{Path, PathBuf};
use std::{env};
use coker::editor::open_editor;
use coker::errors::CokerError;
use coker::project::Project;
use coker::transpiler::*;

const WORKSHEET_SUBDIR: &str = "worksheets";

#[derive(Parser, Debug)]
#[command(name = "coker")]
#[command(about="Edit, analyse and build system models.", long_about = None)]
struct CokerCli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Edit { project: Option<OsString> },
    //    #[command(arg_required_else_help = true)]
    //    Run {
    //        worksheet: String,
    //    },
    Build(BuildArgs),
}

#[derive(Debug, Args)]
struct BuildArgs {
    project: Option<OsString>,
    target: Option<String>,
}

fn main() -> Result<(), CokerError> {
    let args = CokerCli::parse();

    match args.command {
        Commands::Edit { project } => {
            let root = {
                if let Some(path) = project {
                    PathBuf::from(path)
                } else {
                    env::current_dir()?
                }
            };
            let project = validate_project_root(root.as_path())?;
            
            open_editor(project)?;
        }

        Commands::Build(build_args) => {
            let root = {
                if let Some(path) = build_args.project {
                    PathBuf::from(path)
                } else {
                    env::current_dir()?
                }
            };
            let project= validate_project_root(root.as_path())?;
            
            if let Some(target) = build_args.target {
                compile_one(project, &target)?;
            } else {
                compile_all(project)?;
            }
        } 
    }
    Ok(())
}

fn validate_project_root(path: &Path) -> Result<Project, CokerError> {
    
    let project = Project::load(path)?;
    Ok(project)
    
}
