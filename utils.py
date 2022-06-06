import numpy as np
import torch


class Permutator:
    def __init__(self, num_tasks):
        im_size = 28 * 28
        self.perms = [np.identity(im_size)]
        for _ in range(1, num_tasks):
            perm = np.identity(im_size)
            np.random.shuffle(perm)
        self.perms.append(perm)

    def permute_image(self, img, task_id):
        perm_img = np.matmul(self.perms[task_id], np.ravel(img.numpy()))
        return torch.from_numpy(np.reshape(perm_img, img.shape))

    def permute_batch(self, batch, task_id):
        """Permute a batch"""
        X, y = batch
        return (
            torch.vstack(
                [self.permute_image(img, task_id).unsqueeze(0) for img in X]
            ).float(),
            y,
        )

