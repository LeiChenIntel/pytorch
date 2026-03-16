
# Call Stack: `nn.Module` → FX Graph

## Overview

```
torch.compile(model)                         [eval_frame.py]
  → _optimize()
  → get_compiler_fn("inductor")
  → _TorchDynamoContext.__init__()           stores callback (inductor fn)
  → _TorchDynamoContext.__call__(mod)        wraps the nn.Module
      → OptimizedModule(mod, self)           ← NOT an FX graph yet!

compiled_model(inputs)                       ← FIRST CALL triggers compilation
  → OptimizedModule.__call__()
  → OptimizedModule.forward()               (set in _initialize())
      → compile_wrapper()                   [inside __call__ of _TorchDynamoContext]
          → set_eval_frame(callback)        installs CPython hook
          → fn(*args, **kwargs)             runs original forward
              ↓ CPython hook fires on each Python frame
          → CatchErrorsWrapper.__call__()   [convert_frame.py]
          → ConvertFrame.__call__()
          → ConvertFrameAssert.__call__()
          → _compile()
              → compile_inner()
                  → _compile_inner()
                      → compile_frame()
                          → transform_code_object(code, trace_frame)
                              → trace_frame()
                                  → InstructionTranslator(...)   ← bytecode tracer
                                  → tracer.run()                 ← symbolic execution
                                  → DynamoTracerOutput(tracer)   ← FX graph produced HERE
                      → dynamo_output.build_guards()
                      → GuardedCode(out_code, ...)
          → backend(gm, example_inputs)     ← FX GraphModule → inductor
```

---

## Stage-by-Stage Breakdown

### Stage 1: `torch.compile(model)` — Setup (eval_frame.py)

```python
# eval_frame.py

torch.compile(model, backend="inductor")
  → optimize(backend="inductor")
  → _optimize(rebuild_ctx, backend)
      → get_compiler_fn("inductor")          # resolves "inductor" string → callable
      → _optimize_catch_errors(...)
          → OptimizeContext.__init__(callback)  # stores backend, fullgraph, dynamic
      → ctx.__call__(mod)                       # _TorchDynamoContext.__call__
          fn = innermost_fn(fn)                 # unwrap nested compiles
          isinstance(fn, torch.nn.Module) → True
          → new_mod = OptimizedModule(mod, self)
          return new_mod                     # ← returned to user, NOT an FX graph
```

### Stage 2: `_initialize()` in OptimizedModule — Wraps forward (eval_frame.py)

```python
# eval_frame.py: OptimizedModule._initialize()

self.forward = self.dynamo_ctx(self._orig_mod.__call__)
#                  ↑
#   dynamo_ctx.__call__(orig_mod.__call__)
#   returns compile_wrapper function (closure around callback + fn)
```

`self.forward` is now `compile_wrapper`, a closure that:
- calls `set_eval_frame(callback)` to install the CPython hook
- then runs `fn(*args, **kwargs)` — i.e. the original `forward`

### Stage 3: First call `compiled_model(inputs)` — Hook fires (eval_frame.py)

```python
# eval_frame.py: compile_wrapper()

prior = set_eval_frame(None)
_maybe_set_eval_frame(_callback_from_stance(callback))  # ← CPython hook installed
try:
    return fn(*args, **kwargs)   # runs original forward
    # every Python frame encountered triggers the callback
finally:
    set_eval_frame(None)
    _maybe_set_eval_frame(prior)
```

### Stage 4: `CatchErrorsWrapper.__call__()` — Gate (convert_frame.py)

```python
# convert_frame.py: CatchErrorsWrapper.__call__(frame, cache_entry, frame_state)

# Checks: skip files? already traced? dynamo disabled?
if is_skipfile or has_started_execution or config.disable:
    return ConvertFrameReturn()   # skip this frame

# Passes to ConvertFrame
result = self._torchdynamo_orig_backend(
    frame, cache_entry, self.hooks, frame_state, skip=1
)
```

### Stage 5: `ConvertFrame.__call__()` — Tracking + error wrapping (convert_frame.py)

```python
# convert_frame.py: ConvertFrame.__call__()

input_codes.add(frame.f_code)       # track seen frames
result = self._inner_convert(       # → ConvertFrameAssert.__call__()
    frame, cache_entry, hooks, frame_state, skip=skip+1
)
```

### Stage 6: `ConvertFrameAssert.__call__()` — Pre-checks + calls `_compile` (convert_frame.py)

```python
# convert_frame.py: ConvertFrameAssert.__call__()

# Safety checks: generators, __setattr__, no tensor in frame, etc.
if is_generator(code): unimplemented(...)
if not has_tensor_in_frame(frame): return ConvertFrameReturn()

compile_id = get_compile_id(frame_state)   # e.g. [0/0]

result = _compile(
    frame.f_code,
    frame.f_globals, frame.f_locals, ...
    self._torchdynamo_orig_backend,         # inductor backend fn
    ...
    compile_id=compile_id,
)
```

### Stage 7: `_compile()` → `compile_inner()` → `_compile_inner()` (convert_frame.py)

```python
# convert_frame.py: _compile_inner()

log_bytecode("ORIGINAL BYTECODE", ...)   # logs original Python bytecode

dynamo_output = compile_frame(           # → transform_code_object + trace_frame
    code, globals, locals, ...
    compiler_fn,                         # inductor
    ...
)

out_code    = dynamo_output.bytecode      # modified Python bytecode
tracer_output = dynamo_output.tracer_output

check_fn = dynamo_output.build_guards(code, hooks=hooks)
guarded_code = GuardedCode(out_code, check_fn.guard_manager, compile_id, ...)

log_bytecode("MODIFIED BYTECODE", ...)   # logs compiled bytecode
```

### Stage 8: `trace_frame()` + `InstructionTranslator` — FX graph created HERE (convert_frame.py)

```python
# convert_frame.py: trace_frame()

tracer = InstructionTranslator(    # symbolic bytecode interpreter
    instructions,
    code,
    locals, globals, builtins, closure,
    compiler_fn,                   # inductor backend
    ...
)

with tracing(tracer.output.tracing_context), tracer.set_current_tx():
    tracer.run()                   # ← symbolically executes Python bytecode
                                   #   every tensor op → recorded as FX node

tracer_output = DynamoTracerOutput(tracer)
output = tracer_output.output_graph   # ← this IS the FX GraphModule
```

`InstructionTranslator.run()` walks every Python bytecode instruction. When it sees a tensor operation (e.g. `conv2d`, `relu`), it records it as an `fx.Node` instead of executing it. The result is a `torch.fx.GraphModule`.

### Stage 9: Backend called with FX Graph → Inductor

```python
# Inside output_graph.py / InstructionTranslator

gm = torch.fx.GraphModule(...)    # the FX graph
example_inputs = [...]

# compiler_fn is the inductor backend (wrapped by WrapBackendDebug)
compiled_fn = compiler_fn(gm, example_inputs)
# → inductor lowers FX graph to Triton/C++ kernel
```

---

## Summary Table

| Step | File | What happens |
|---|---|---|
| `torch.compile(model)` | `eval_frame.py` | Creates `OptimizedModule`, stores backend |
| `compiled_model(x)` first call | `eval_frame.py` | `compile_wrapper` installs CPython hook |
| CPython hook fires | `eval_frame.py → convert_frame.py` | `set_eval_frame(callback)` routes frames |
| `CatchErrorsWrapper.__call__` | `convert_frame.py` | Skip checks (skipfiles, already traced) |
| `ConvertFrameAssert.__call__` | `convert_frame.py` | Safety checks, calls `_compile()` |
| `_compile_inner()` | `convert_frame.py` | Logs bytecode, calls `compile_frame()` |
| `trace_frame()` | `convert_frame.py` | Creates `InstructionTranslator` |
| `tracer.run()` | `symbolic_convert.py` | **FX graph built here** — symbolic execution |
| `compiler_fn(gm, inputs)` | `inductor/compile_fx.py` | FX graph → compiled kernel |
