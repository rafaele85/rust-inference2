    use crate::activation::gelu;
    use crate::attention::extract_v;
    use crate::linear::{normalize_and_project, project_and_residual, NormalizeAndProject, ProjectAndResidualArgs};

    pub struct TransformerPassArgs<'a> {
        pub hidden: &'a mut Vec<f32>,
        pub wte: &'a Vec<f32>,
        pub wpe: &'a Vec<f32>,
        pub layer_weights: &'a (
          Vec<f32>, Vec<f32>,Vec<f32>, Vec<f32>,Vec<f32>, Vec<f32>,Vec<f32>, Vec<f32>,Vec<f32>, Vec<f32>,Vec<f32>, Vec<f32>,
        ),
        pub norm_final_weight: &'a Vec<f32>,
        pub norm_final_bias: &'a Vec<f32>,
        pub dim: usize,
    }

    pub fn transformer_pass(args: TransformerPassArgs) {
        let ffn_dim = 3072;
        let qkv_combined_dim = args.dim * 3; // all 3 weights are of the same size

        let (ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b, ln2_w, ln2_b, ffn_up_w, ffn_up_b, ffn_down_w, ffn_down_b) = args.layer_weights;

        let qkv = normalize_and_project(NormalizeAndProject {hidden: args.hidden, norm_weights: &ln1_w, norm_bias: &ln1_b, projection_weights: &qkv_w, projection_bias: &qkv_b, projection_dim: qkv_combined_dim});

        let v = extract_v(&qkv);

        project_and_residual(ProjectAndResidualArgs {activation_weights: &v, projection_weights: &proj_w, projection_bias: &proj_b, dim: args.dim, hidden: args.hidden });

        let ffn_up = normalize_and_project(NormalizeAndProject {hidden: args.hidden, norm_weights: &ln2_w, norm_bias: &ln2_b, projection_weights: &ffn_up_w, projection_bias: &ffn_up_b, projection_dim: ffn_dim});

        let ffn_act = gelu(&ffn_up);

        project_and_residual(ProjectAndResidualArgs {activation_weights: &ffn_act, projection_weights: &ffn_down_w, projection_bias: &ffn_down_b, dim: args.dim, hidden: args.hidden });
    }