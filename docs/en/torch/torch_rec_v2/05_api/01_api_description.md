# API Description

The Rec SDK Torch are developed using Python interfaces. From a code invocation perspective, you can call all Python interfaces. This section lists only the external interfaces that the service provides. Do not directly call other interfaces that are not described. Rec SDK Torch is an extension based on the TorchRec APIs. Rec SDK Torch APIs depend on the classes and methods provided by TorchRec, but they do not support every TorchRec feature. Therefore, this section describes the TorchRec API limitations that apply when you build models with Rec SDK Torch.

> [!NOTE]NOTE
> Because native TorchRec APIs are outside the management scope of Rec SDK Torch, parameter validity checks are not performed on the native TorchRec APIs used by the following APIs. You must ensure that the parameters are correct.
