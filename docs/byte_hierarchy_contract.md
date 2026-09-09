# Byte hierarchy contract — version 1

Scope: executable byte hierarchy only, against the accepted inventory. The 1,929,579 full structural candidate remains an explicit smoke-stage exception at −3.52105%; the configured 2% tolerance and resolved snapshot are unchanged. Mamba remains absent; only the non-shared components can be reconciled as executable.

## Symbols and padding

The canonical registry remains configs' CONTROL_IDS, also recorded in the resolved snapshot: bytes 0..255; PAD=256; BOS=257; EOS=258; tool_call=259; tool_result=260; rag_begin=261; rag_end=262; code_begin=263; code_end=264; error=265; retry=266. Total vocabulary 267 (256 payload values, one padding value, ten non-padding controls). Name/ID helpers derive from that registry; they do not allocate IDs.

Raw bytes are never decoded or normalized. Converting a mixed symbol stream directly to payload bytes rejects controls. BOS is out-of-band request conditioning and cannot be consumed as a payload slot. PAD is storage only and cannot be consumed or admitted to the completed-patch encoder. Other controls, including EOS, occupy a symbol slot when consumed. The caller owns EOS stopping and later tool/protocol interpretation; this layer does not execute or interpret tools. Predictions have PAD/BOS logits masked to negative infinity intentionally; other logits must remain finite. No sampling policy is introduced.

Explicit padding returns a right-padded symbol tensor plus exact per-row lengths. It does not complete an observed patch. Complete observed count is floor(L/8); padded storage width uses ceil(L/8)*8. Rows with different lengths use separate streaming states (or caller grouping at the same phase), not PAD as a recurrent input.

## Executable components

One ByteEmbedding owns the 267-entry symbol table and the eight-entry position table, shared by encoding and decoding without duplicate registration. PAD has a zero embedding row and does not contribute embedding gradients. The encoder consumes only completed [B,N,8] symbol patches through the hierarchy API: embedding + local position → left-padded depthwise kernel-3 convolution → linear gate (value * SiLU(gate)) → RMSNorm → mean over all eight local positions → projection to d_model. It sees a complete patch only after that patch is observed; its output cannot condition predictions within that same patch.

The local decoder starts hidden state h0=tanh(context_projection(c)). For prediction x_j it emits the unified head over RMSNorm(h_j). Only after observing x_j does it update h_(j+1)=GRUCell(embedding(x_j)+position(j),h_j). Teacher forcing shifts this dependency explicitly; the last target is never read to predict itself. There is no Transformer machinery or recurrent global substitute.

## Bootstrap and future trunk boundary

initial_conditioning(batch) = learned BOS projection of the shared BOS symbol embedding. start(batch) uses that conditioning for the first local hidden state. A caller may supply an explicit trusted initial conditioning tensor (for future BOS-conditioned shared processing); it must contain only context available before the first payload symbol. This is a semantic caller obligation that tensor checks cannot prove.

consume(symbol,state) returns a fresh state and either no patch event or one completed-patch event. Before eight observed symbols there is no event. At the eighth symbol it emits [B,d_model] and increments completed_patches. The local state then has hidden=None, meaning **awaiting external trunk conditioning**, not a fabricated recurrent state. predict/consume refuse to proceed until condition(context,state) receives a real [B,d_model] context from the caller. No shared state, fallback recurrent model, identity trunk or callback wrapper is implemented. A future Mamba integration will own the global state and consume each event exactly once.

States are synchronous batches: pending IDs [B,0..7], real local hidden [B,decoder_dim] or an explicit awaiting marker, and the number of completed emitted patches. Zero-length pending tensors represent an empty byte prefix; no zero hidden tensor is used as accidental bootstrap. Methods do not mutate incoming state tensors. Saved state dictionaries include schema, resolved-config hash, hidden/pending tensors and phase/count. Restore validates schema, IDs, shapes, dtype/device, finite hidden values and phase invariants. Serialization is an inference-state snapshot detached from autograd, not a training checkpoint; use the same model weights when restoring. At an awaiting boundary the caller must also persist the outbound completed-patch event and external conditioning/trunk state. Restoring the local hierarchy alone cannot manufacture missing global context.

## Validation gates

FP32 CPU tolerance: atol=1e-6, rtol=1e-5 for full/step or independent-equation comparisons. Same-prefix/same-state adversarial tests also record maximum absolute error. Tests use an explicitly test-only causal accumulator of completed patch outputs to exercise multi-patch boundaries; that accumulator is not production code or evidence of Mamba behavior. Independent GRU/RMSNorm equations, gradient dependency checks, completion counts, cross-patch encoder isolation and serialized continuation provide separate oracles.

Reconcile all five implemented parameter groups (embedding, encoder, decoder, output, bootstrap) against inventory v1, including per-tensor shapes. Shared parameters remain structural only. Any mismatch blocks integration. These are clarifications of Bible §§6–9/17/81/136/168 and the accepted inventory, not modifications to either FINAL specification. AC-001..005 remain in force; same-chunk MoE causality remains deferred.

Implementation review: an initial last-position aggregation failed the all-eight-position gradient gate (zero derivative for positions 0..4). It was replaced by mean pooling of all eight causally mixed positions, with no added/removed parameters. The FINAL specifications do not prescribe the faulty endpoint aggregation; this is an implementation correction, not a Bible change. Independent convolution/gate/norm equations cover the corrected pooling.
