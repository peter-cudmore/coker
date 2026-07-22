use coker_bytecode::{BytecodeModule, EvaluateLayer, InputSpec, Layer, OutputSpec, Program};

use crate::{
    lower::{
        compile_bilinear_layer, compile_evaluate_input_binding, compile_evaluate_output_binding,
        compile_generic_layer,
    },
    model::{CompileContext, ExportedLayer, ExportedModule, ExportedProgram},
    util::{checked_add_u32, checked_u16, checked_u8_length, required_field},
    CompileError,
};

pub(crate) fn compile_exported_module(
    exported_module: ExportedModule,
) -> Result<BytecodeModule, CompileError> {
    let exported_programs = index_exported_programs(exported_module)?;
    let function_count = exported_programs.len();
    let mut compile_context = CompileContext {
        compiled_programs: vec![None; function_count],
        visiting: vec![false; function_count],
        exported_programs,
    };

    let mut functions = Vec::with_capacity(function_count);
    for function_id in 0..function_count {
        functions.push(compile_context.compile_function(function_id as u16)?);
    }
    Ok(BytecodeModule::new(functions))
}

fn index_exported_programs(
    exported_module: ExportedModule,
) -> Result<Vec<ExportedProgram>, CompileError> {
    if exported_module.functions.is_empty() {
        return Err(CompileError::InvalidField {
            field: "functions",
            reason: "expected at least one function",
        });
    }

    let function_count = exported_module.functions.len();
    let mut indexed_programs = vec![None; function_count];
    for exported_function in exported_module.functions {
        let function_id = checked_u16(exported_function.function_id, "function_id")?;
        let function_index = function_id as usize;
        if function_index >= function_count {
            return Err(CompileError::InvalidField {
                field: "function_id",
                reason: "expected dense ids from 0 to function_count - 1",
            });
        }
        if indexed_programs[function_index].is_some() {
            return Err(CompileError::InvalidField {
                field: "function_id",
                reason: "duplicate function id",
            });
        }
        indexed_programs[function_index] = Some(exported_function.program);
    }

    indexed_programs
        .into_iter()
        .map(|program| {
            program.ok_or(CompileError::InvalidField {
                field: "function_id",
                reason: "expected dense ids from 0 to function_count - 1",
            })
        })
        .collect()
}

impl CompileContext {
    fn compile_function(&mut self, function_id: u16) -> Result<Program, CompileError> {
        let function_index = function_id as usize;
        if let Some(compiled_program) = self.compiled_programs[function_index].clone() {
            return Ok(compiled_program);
        }
        if self.visiting[function_index] {
            return Err(CompileError::NotImplemented(
                "recursive function evaluation".to_string(),
            ));
        }

        self.visiting[function_index] = true;
        let exported_program = self.exported_programs[function_index].clone();
        let compiled_program = self.compile_program(function_id, exported_program)?;
        self.visiting[function_index] = false;
        self.compiled_programs[function_index] = Some(compiled_program.clone());
        Ok(compiled_program)
    }

    fn compile_program(
        &mut self,
        function_id: u16,
        exported_program: ExportedProgram,
    ) -> Result<Program, CompileError> {
        let input_specs = exported_program
            .input_layer
            .inputs
            .into_iter()
            .map(|input_spec| {
                let memory = input_spec.memory;
                Ok::<_, CompileError>(InputSpec {
                    workspace_offset: memory.location,
                    length: checked_u16(memory.count, "input.memory.count")?,
                })
            })
            .collect::<Result<Vec<_>, CompileError>>()?;

        let output_specs = exported_program
            .output_layer
            .outputs
            .into_iter()
            .map(|output_spec| {
                let memory = output_spec.memory;
                Ok::<_, CompileError>(OutputSpec {
                    workspace_offset: memory.location,
                    length: checked_u16(memory.count, "output.memory.count")?,
                })
            })
            .collect::<Result<Vec<_>, CompileError>>()?;

        let workspace_size = exported_program.workspace.count;
        let mut required_workspace_size = workspace_size;
        let intermediate_layers = exported_program
            .intermediate_layers
            .into_iter()
            .map(|layer| self.compile_layer(layer, &mut required_workspace_size))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Program::new(
            function_id,
            workspace_size,
            required_workspace_size,
            input_specs,
            output_specs,
            intermediate_layers,
        ))
    }

    fn compile_layer(
        &mut self,
        exported_layer: ExportedLayer,
        required_workspace_size: &mut u32,
    ) -> Result<Layer, CompileError> {
        match exported_layer.kind.as_str() {
            "bilinear" => compile_bilinear_layer(exported_layer).map(Layer::Bilinear),
            "generic" => compile_generic_layer(exported_layer).map(Layer::Generic),
            "evaluate" => self
                .compile_evaluate_layer(exported_layer, required_workspace_size)
                .map(Layer::Evaluate),
            unsupported_kind => Err(CompileError::NotImplemented(format!(
                "layer kind {unsupported_kind}"
            ))),
        }
    }

    fn compile_evaluate_layer(
        &mut self,
        exported_layer: ExportedLayer,
        required_workspace_size: &mut u32,
    ) -> Result<EvaluateLayer, CompileError> {
        let callee_function_id = checked_u16(
            required_field(exported_layer.callee_function_id, "callee_function_id")?,
            "callee_function_id",
        )?;
        let callee_program = self.compile_function(callee_function_id)?;

        let exported_inputs = required_field(exported_layer.inputs, "inputs")?;
        let exported_outputs = required_field(exported_layer.outputs, "outputs")?;
        if exported_inputs.len() != callee_program.input_specs.len() {
            return Err(CompileError::InvalidField {
                field: "inputs",
                reason: "evaluate input binding count does not match callee inputs",
            });
        }
        if exported_outputs.len() != callee_program.output_specs.len() {
            return Err(CompileError::InvalidField {
                field: "outputs",
                reason: "evaluate output binding count does not match callee outputs",
            });
        }

        let input_bindings = exported_inputs
            .into_iter()
            .zip(callee_program.input_specs.iter())
            .map(|(binding, input_spec)| compile_evaluate_input_binding(binding, input_spec.length))
            .collect::<Result<Vec<_>, _>>()?;
        let output_bindings = exported_outputs
            .into_iter()
            .zip(callee_program.output_specs.iter())
            .map(|(binding, output_spec)| {
                compile_evaluate_output_binding(binding, output_spec.length)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let scratch_offset = *required_workspace_size;
        *required_workspace_size = checked_add_u32(
            scratch_offset,
            callee_program.required_workspace_size,
            "scratch_offset",
        )?;

        Ok(EvaluateLayer {
            scratch_offset,
            callee_function_id,
            input_count: checked_u8_length(input_bindings.len(), "inputs")?,
            output_count: checked_u8_length(output_bindings.len(), "outputs")?,
            input_bindings,
            output_bindings,
        })
    }
}
