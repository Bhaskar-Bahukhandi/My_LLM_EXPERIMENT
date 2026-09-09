# Parameter inventory contract, version 1

This tranche deliberately stops before implementing the byte encoder, local decoder or Mamba model. Exact instantiated auditing therefore counts a **declared structural inventory**, not a functioning model. It uses real torch.nn.Parameter objects on device=meta and exposes no inference API. No weights are trained, initialized for learning, or claimed to generate bytes. The next tranche must compare the inventory's named tensor shapes and component counts against the executable model; a discrepancy invalidates this candidate's parameter readiness.

The proposed dense topology follows Bible §§8/9/17/19, which leave the local mixer and decoder implementation choices open. ENGINEERING PREFERENCE: use a small causal depthwise kernel-3 patch mixer, a gated pointwise projection and RMSNorm; a GRU-cell local decoder with context initialization; one shared byte embedding and one position-in-patch table. These are a testable parameter-budget contract for the next tranche, not evidence of learning quality. Separate learned BOS projection is included. No unused padding matrices are added to hit the target.

| Component | Declared tensors |
|---|---|
| embedding | symbol table [267,byte_dim], patch positions [patch_size,byte_dim] |
| encoder | depthwise [byte_dim,1,3] + bias; gate [2*byte_dim,byte_dim] + bias; norm [byte_dim]; projection [d_model,byte_dim] + bias |
| shared | per layer: pre-norm [d_model]; official subset in projection [2*d_inner+2*d_state+nheads,d_model]; depthwise [d_inner+2*d_state,1,d_conv]+bias; dt_bias/A_log/D [nheads]; gated norm [d_inner]; output [d_model,d_inner] |
| decoder | context [decoder_dim,d_model]+bias; GRU input [3*decoder_dim,byte_dim], recurrent [3*decoder_dim,decoder_dim], two biases [3*decoder_dim]; norm [decoder_dim] |
| output | untied unified head [267,decoder_dim]+bias |
| bootstrap | [d_model,byte_dim]+bias |

Versioned controls: raw 0..255, pad=256, bos=257, eos=258, tool_call=259, tool_result=260, rag_begin=261, rag_end=262, code_begin=263, code_end=264, error=265, retry=266. This tranche records the registry only; byte/control framing and PAD state transitions must be defined and tested in the next tranche. No tokenizer or protocol implementation is present.

Search policy enumerates only explicitly authored width/depth choices, with widths divisible by 64 as a conservative clean-candidate policy (not a mathematical Mamba requirement). Backend mathematical legality is tested separately. Fixed dimensions remain fixed; no tensor is enlarged to reach a target. Exact counts come from instantiated inventories; formulas are cross-checks only. If no legal clean inventory is within tolerance, return OUTSIDE_TARGET with the closest candidate. No result is training authorization.

Preflight reports known FP32 weight/gradient/Adam-moment payload and recurrent/convolution/local-state planning bytes. Activations, allocator overhead, transient buffers, optimizer step overhead and process RSS remain unknown. Known lower-bound overflow rejects a candidate; passing that bound returns UNKNOWN_REQUIRES_MEASUREMENT. Dense total and active inventory parameters coincide; runtime activity is not measured. Metadata-only construction uses negligible tensor storage and proves no host/device memory fit.
