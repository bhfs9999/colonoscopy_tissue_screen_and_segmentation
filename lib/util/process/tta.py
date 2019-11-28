import torch

ttas = {
    'origin': lambda img: img,
    'rot90': lambda img: torch.rot90(img, k=1, dims=[2, 3]),
    'rot180': lambda img: torch.rot90(img, k=2, dims=[2, 3]),
    'rot270': lambda img: torch.rot90(img, k=3, dims=[2, 3]),
    'flip_h': lambda img: torch.flip(img, dims=[3]),
    # 'flip_h_rot90': lambda img: torch.rot90(torch.flip(img, dims=[3]), k=1, dims=[2, 3]),
    'flip_h_rot180': lambda img: torch.rot90(torch.flip(img, dims=[3]), k=2, dims=[2, 3]),
    # 'flip_h_rot270': lambda img: torch.rot90(torch.flip(img, dims=[3]), k=3, dims=[2, 3]),
}

dettas = {
    'origin': lambda img: img,
    'rot90': lambda img: torch.rot90(img, k=-1, dims=[2, 3]),
    'rot180': lambda img: torch.rot90(img, k=-2, dims=[2, 3]),
    'rot270': lambda img: torch.rot90(img, k=-3, dims=[2, 3]),
    'flip_h': lambda img: torch.flip(img, dims=[3]),
    'flip_h_rot90': lambda img: torch.flip(torch.rot90(img, k=-1, dims=[2, 3]), dims=[3]),
    'flip_h_rot180': lambda img: torch.flip(torch.rot90(img, k=-2, dims=[2, 3]), dims=[3]),
    'flip_h_rot270': lambda img: torch.flip(torch.rot90(img, k=-3, dims=[2, 3]), dims=[3]),
}

if __name__ == '__main__':
    a = torch.randn(4, 3, 100, 200)
    for name, tta in ttas.items():
        b = tta(a)
        print(b.shape)
