<h1>FlexAI is a high-level deep learning package intended to facilitate training PyTorch models combining ease and flexibility.</h1>
Mainly based on the idea of a <b>Learner</b> which automate the training and validation process. Instead of having to write training loops, the user should only be concerned with choosing their model architecture, optimizer, metrics, data augmentation techniques.</br>
To establish flexibility, the learner supports the use of <b>callbacks</b>. A callback can be called at on_init which is called whenever the learner object is initialized, before_fit and after_fit that are called before and after a training/validation loop, before_epcoh and after_epcoh before training/validation epoch, or before_batch and after_batch before and after training/validating a dataloader batch.</br></br>
The package provides a collection of basic callbacks like LoggerCB which is prints metric values at the end of every epoch, LRFinderCB used for choosing a suitable learning rate, ForwardHookCB which registers a forward hook to the modules of the model, and many other callbacks.</br></br>
The package also provides a few custom modules based on PyTorch modules. DenseLayer which is fundamentally a PyTorch Linear layer with an activation layer on top, ConvLayer is a Conv2d layer with an activation on top, ResLayer a residual layer. The goal of making these modules is to make things like adding a norm layer or initializing weights cleaner and more efficient.</br></br>
There is a vision sub-module that contains transforms and utility functions that can be helpful when working on a computer vision task.