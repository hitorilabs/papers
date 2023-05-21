# Preprocessing Techniques

On the surface, pre-processing is quite straightforward.
All we're doing to taking data and transforming it into
some format we want (i.e. binary, chunks, numerical,
etc.). 

However, the problem is that it's such a simple idea that
everyone just rolls their own pre-processing method. I
don't think we should standardize it, but we should all
have some idea of the most elegant path.

![image](https://github.com/hitorilabs/papers/assets/131238467/9597fe82-0d20-4af7-bf91-acfff08d68d9)


https://gist.github.com/ZijiaLewisLu/eabdca955110833c0ce984d34eb7ff39?permalink_comment_id=3417135

just use `np.memmap` for everything - it's all tensors.