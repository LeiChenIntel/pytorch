import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSoftmaxModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvSoftmaxModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 7, 7]
        x = x.view(x.size(0), -1)  # [B, 64*7*7]
        x = self.fc(x)  # [B, num_classes]
        x = F.softmax(x, dim=1)
        return x


if __name__ == "__main__":
    model = ConvSoftmaxModel(num_classes=10)
    model.eval()

    # Dummy input: batch=1, channels=1, height=28, width=28 (MNIST-like)
    dummy_input = torch.randn(1, 1, 28, 28)

    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sum per sample: {output.sum(dim=1)}")  # Should be ~1.0

    # --- Export to ONNX (best for Netron) ---
    onnx_path = "conv_softmax_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"ONNX model saved to: {onnx_path}")

    # --- Optional: also save as TorchScript (.pt) for Netron ---
    # scripted_model = torch.jit.trace(model, dummy_input)
    # pt_path = "conv_softmax_model.pt"
    # scripted_model.save(pt_path)
    # print(f"TorchScript model saved to: {pt_path}")

    # --- Export to ONNX using dynamo (PyTorch 2.5+ recommended) ---
    # This internally uses torch.compile / TorchDynamo tracing
    dynamo_onnx_path = "conv_softmax_dynamo.onnx"
    torch.onnx.export(
        model,
        (dummy_input,),
        dynamo_onnx_path,
        dynamo=True,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"Dynamo ONNX model saved to: {dynamo_onnx_path}")

    # --- Run ONNX model with ONNX Runtime ---
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(dynamo_onnx_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        ort_output = sess.run(None, {input_name: dummy_input.numpy()})
        print(f"ORT output shape: {ort_output[0].shape}")
        print(f"ORT sum per sample: {ort_output[0].sum(axis=1)}")  # Should be ~1.0
    except ImportError:
        print("onnxruntime not installed, skipping ORT inference. pip install onnxruntime")
