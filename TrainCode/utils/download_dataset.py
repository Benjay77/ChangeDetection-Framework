import os.path

import gdown

url = 'https://drive.google.com/uc?id=1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9'
output = 'data/ChangeDetectionDataset.rar'

if not os.path.isdir('data/'):
    os.mkdir('data/')
gdown.download(url, output, quiet=False)
