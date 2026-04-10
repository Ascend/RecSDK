# 接口说明<a name="ZH-CN_TOPIC_0000002302229620"></a>

Rec SDK Torch通过Python接口进行应用开发，从代码调用角度上来说所有Python侧接口都可以被调用。本章节仅列出业务提供的对外接口，其余未进行说明的接口用户请勿直接调用。Rec SDK Torch是基于TorchRec接口的扩展，Rec SDK Torch的接口依赖于TorchRec提供的类和方法，但是并不能支持TorchRec的所有功能。所以本章节会介绍基于Rec SDK Torch搭建模型时使用的TorchRec的接口的限制。

>[!NOTICE]
>由于TorchRec原生接口不在Rec SDK Torch的管理范围内，因此将不会对后续接口中涉及到的TorchRec原生接口做参数合法性校验，需用户自行保证参数正确性。
