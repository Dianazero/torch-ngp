if __name__ == '__main__':
     
    import torch
 
    x = torch.Tensor([[1,2], [2,3], [3,4]])
    print(x.size()[0])
 
    print(x)
    d=x.unsqueeze(1).expand(3,2, 2)
    print(d)