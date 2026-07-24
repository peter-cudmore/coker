use coker_bytecode::decode_qp_program;
use osqp::{CscMatrix, Problem, Settings};

use crate::{Module, ModuleBuilder, RuntimeError};

#[derive(Debug, Clone)]
pub struct QpSolveResult {
    pub solution: Vec<f64>,
    pub success: bool,
    pub status: String,
}

pub struct QpRuntime {
    n: usize,
    m: usize,
    parameter_lengths: Vec<usize>,
    p_rows: Vec<usize>,
    p_cols: Vec<usize>,
    a_rows: Vec<usize>,
    a_cols: Vec<usize>,
    coefficient_lengths: Vec<usize>,
    evaluator: Module,
    warm_start: bool,
}

fn empty_csc(nrows: usize, ncols: usize) -> CscMatrix<'static> {
    CscMatrix {
        nrows,
        ncols,
        indptr: vec![0; ncols + 1].into(),
        indices: Vec::new().into(),
        data: Vec::new().into(),
    }
}

impl QpRuntime {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RuntimeError> {
        let archive = decode_qp_program(bytes)?;
        if archive.p_rows.len() != archive.p_cols.len()
            || archive.a_rows.len() != archive.a_cols.len()
            || archive.coefficient_lengths.len() != 6
        {
            return Err(RuntimeError::Validation("invalid QP archive structure".to_string()));
        }
        Ok(Self {
            n: archive.n as usize,
            m: archive.m as usize,
            parameter_lengths: archive.parameter_lengths.into_iter().map(|v| v as usize).collect(),
            p_rows: archive.p_rows.into_iter().map(|v| v as usize).collect(),
            p_cols: archive.p_cols.into_iter().map(|v| v as usize).collect(),
            a_rows: archive.a_rows.into_iter().map(|v| v as usize).collect(),
            a_cols: archive.a_cols.into_iter().map(|v| v as usize).collect(),
            coefficient_lengths: archive.coefficient_lengths.into_iter().map(|v| v as usize).collect(),
            evaluator: ModuleBuilder::new_from_bytes(&archive.coefficient_program)?.build()?,
            warm_start: archive.warm_start,
        })
    }

    pub fn solve(&mut self, parameters: &[&[f32]], warm_start: Option<&[f64]>) -> Result<QpSolveResult, RuntimeError> {
        if parameters.len() != self.parameter_lengths.len()
            || parameters.iter().zip(&self.parameter_lengths).any(|(value, length)| value.len() != *length)
        {
            return Err(RuntimeError::Validation("QP parameter dimensions do not match archive".to_string()));
        }
        let output_len: usize = self.evaluator.info().output_specs.iter().map(|spec| spec.length as usize).sum();
        let mut output = vec![0.0_f32; output_len];
        let inputs = self.evaluator.validate_inputs(parameters)?;
        let outputs = self.evaluator.validate_outputs(&mut output)?;
        self.evaluator.execute(inputs, outputs);

        let mut offset = 0;
        let take = |length: usize, values: &[f32], offset: &mut usize| -> Result<Vec<f64>, RuntimeError> {
            if *offset + length > values.len() { return Err(RuntimeError::Validation("QP evaluator output too short".to_string())); }
            let result = values[*offset..*offset + length].iter().map(|value| *value as f64).collect();
            *offset += length;
            Ok(result)
        };
        let px = take(self.coefficient_lengths[0], &output, &mut offset)?;
        let q = take(self.coefficient_lengths[1], &output, &mut offset)?;
        let ax = take(self.coefficient_lengths[2], &output, &mut offset)?;
        let l = take(self.coefficient_lengths[3], &output, &mut offset)?;
        let u = take(self.coefficient_lengths[4], &output, &mut offset)?;

        let mut p = vec![vec![0.0_f64; self.n]; self.n];
        for ((&row, &col), value) in self.p_rows.iter().zip(&self.p_cols).zip(px) { p[row][col] = value; }
        let mut a = vec![vec![0.0_f64; self.n]; self.m];
        for ((&row, &col), value) in self.a_rows.iter().zip(&self.a_cols).zip(ax) { a[row][col] = value; }
        let settings = Settings::default().verbose(false);
        let a_matrix = if self.m == 0 {
            empty_csc(0, self.n)
        } else {
            CscMatrix::from(&a[..])
        };
        let mut problem = Problem::new(CscMatrix::from(&p[..]), &q, a_matrix, &l, &u, &settings)
            .map_err(|error| RuntimeError::QpSolver(error.to_string()))?;
        if self.warm_start { if let Some(value) = warm_start { problem.warm_start_x(value); } }
        let status = problem.solve();
        let solution = status.x().map(|value| value.to_vec()).unwrap_or_else(|| vec![f64::NAN; self.n]);
        Ok(QpSolveResult { success: status.x().is_some(), status: format!("{:?}", status), solution })
    }
}
