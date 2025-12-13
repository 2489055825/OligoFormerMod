import pandas as pd
import numpy as np
import random
from torch.utils import data
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import torch
from typing import Optional

try:
    import h5py  # Optional; only required when contact maps are used
except ImportError:
    h5py = None

nucleic_char = ['A', 'U', 'C', 'G', 'X'] # 
structure_char = ['.', '(', ')']

# ACGU 1234

enc_protein = OneHotEncoder().fit(np.array(nucleic_char).reshape(-1, 1))
enc_structure =  LabelEncoder().fit(np.array(structure_char))
def sequence_OneHot(x):
    # Minimal change: map any non-A/U/C/G to 'X' before encoding
    chars = list(x) if not isinstance(x, str) else list(x)
    mapped = [(c.upper() if c.upper() in ('A','U','C','G') else 'X') for c in chars]
    return enc_protein.transform(np.array(mapped).reshape(-1, 1)).toarray().T

def structure_Label(x):
    return enc_structure.transform(np.array(x).reshape(-1, 1)).T

def Embed(RNA):
    nucleotides = 'ACGU' 
    char_to_int = dict((c, i + 1 ) for i, c in enumerate(nucleotides))
    return [char_to_int[i] for i in RNA]


class ContactMapStore:
    """Wrapper around an h5 contact-map file."""
    def __init__(self, path: str):
        if h5py is None:
            raise ImportError("Install h5py to read contact_map files.")
        self.path = path
        self.handle = h5py.File(path, "r")

    def __contains__(self, key: str) -> bool:
        return key in self.handle

    def get(self, key: str):
        if key not in self.handle:
            return None, {}
        ds = self.handle[key]
        attrs = dict(ds.attrs) if ds.attrs else {}
        return ds[()], attrs

    def close(self):
        try:
            self.handle.close()
        except Exception:
            pass


class data_process_loader(data.Dataset):
    def __init__(self, df_index, labels, y, df, dataset,root, contact_map_path: Optional[str] = None, contact_key_col: Optional[str] = None, contact_p_column: Optional[str] = None):
        'Initialization'
        self.labels = labels
        self.df_index = df_index
        self.y = y
        self.df = df
        self.dataset = dataset
        self.siRNA_ref = {}
        self.root = root
        self.contact_map_path = contact_map_path
        self.contact_key_col = contact_key_col
        self.contact_p_column = contact_p_column
        self.contact_store = ContactMapStore(contact_map_path) if contact_map_path else None
        with open(root + '/fasta/' + self.dataset + '_siRNA.fa') as f:
            for line in f:
                if line.startswith('>'):
                    name = line.strip().replace('>','')
                else:
                    seq = line.strip()
                    self.siRNA_ref[seq] = name
        self.mRNA_ref = {}
        with open(root + '/fasta/' + self.dataset + '_mRNA.fa') as f:
            for line in f:
                if line.startswith('>'):
                    name = line.strip().replace('>','')
                else:
                    seq = line.strip()
                    self.mRNA_ref[seq] = name

    def __del__(self): # 如果属性存在，则执行del时关闭属性
        if getattr(self, "contact_store", None) is not None:
            self.contact_store.close()

    def _get_contact_key(self, row) -> str: # 返回 siRNA * mRNA 作为 key，也可自定义一列作为key 
        if self.contact_key_col and self.contact_key_col in row:
            return row[self.contact_key_col]
        return f"{row['siRNA']}x{row['mRNA']}"

    def _get_default_contact_map(self, siRNA_seq: str, mRNA_seq: str): # 全0矩阵，shape正确
        return np.zeros((len(siRNA_seq), len(mRNA_seq)), dtype=np.float32)

    def _get_p_value(self, row, attrs: dict): 
        # 如果 contact_p_column 存在，返回 p 的值
        # 如果不存在，查找dataset内有无("p_value", "p", "pval")作为列
        if self.contact_p_column and self.contact_p_column in row:
            try:
                return float(row[self.contact_p_column])
            except Exception:
                return 0.0
        for key in ("p_value", "p", "pval"):
            if attrs is not None and key in attrs:
                try:
                    return float(attrs[key])
                except Exception:
                    continue
        return 0.0

    def _load_contact_map_and_p(self, row):
        # 取出 p 和 contact_map 返回
        contact_map = None
        attrs = {}
        if self.contact_store is not None:
            key = self._get_contact_key(row)
            contact_map, attrs = self.contact_store.get(key)
        siRNA_seq = row["siRNA"]
        mRNA_seq = row["mRNA"]
        if contact_map is None:
            contact_map = self._get_default_contact_map(siRNA_seq, mRNA_seq)
        contact_map = np.asarray(contact_map, dtype=np.float32)
        p_value = self._get_p_value(row, attrs)
        return contact_map, np.float32(p_value)

    def __len__(self):
        return len(self.df_index)
    def __getitem__(self, index):
        index = self.df_index[index]
        label = float(self.labels[index])
        y = np.int64(self.y[index])
        # siRNA
        siRNA_seq = self.df.iloc[index]['siRNA']
        siRNA_seq = [*siRNA_seq]
        siRNA_seq = sequence_OneHot(siRNA_seq)
        siRNA = np.expand_dims(siRNA_seq, axis=2).transpose([2, 1, 0])
        # mRNA
        mRNA_seq = self.df.iloc[index]['mRNA']
        mRNA_seq = [*mRNA_seq]
        mRNA_seq = sequence_OneHot(mRNA_seq)
        mRNA = np.expand_dims(mRNA_seq, axis=2).transpose([2, 1, 0])
        # siRNA RNA-FM
        siRNA_seq = self.df.iloc[index]['siRNA']
        siRNA_FM = np.load(self.root + '/RNAFM/' + self.dataset + '_siRNA/representations/'+ str(self.siRNA_ref[siRNA_seq]) + '.npy')
        # mRNA RNA-FM
        mRNA_seq = self.df.iloc[index]['mRNA']
        mRNA_FM = np.load(self.root + '/RNAFM/' + self.dataset + '_mRNA/representations/' + str(self.mRNA_ref[mRNA_seq])+'.npy') 
        td = self.df.iloc[index]['td']
        td = torch.tensor([float(i) for i in td.split(',')])
        contact_map, contact_p = self._load_contact_map_and_p(self.df.iloc[index])
        return  siRNA, mRNA, siRNA_FM, mRNA_FM, label, y, td, contact_map, contact_p




class data_process_loader_infer(data.Dataset):
    def __init__(self, df_index, df, dataset, contact_map_path: Optional[str] = None, contact_key_col: Optional[str] = None, contact_p_column: Optional[str] = None):
        self.df_index = df_index
        self.df = df
        self.dataset = dataset
        self.siRNA_ref = {}
        self.contact_map_path = contact_map_path
        self.contact_key_col = contact_key_col
        self.contact_p_column = contact_p_column
        self.contact_store = ContactMapStore(contact_map_path) if contact_map_path else None
        with open('./data/infer/' + self.dataset + '/siRNA.fa') as f:
            for line in f:
                if line.startswith('>'):
                    name = line.strip().replace('>','')
                else:
                    seq = line.strip()
                    self.siRNA_ref[seq] = name
        self.mRNA_ref = {}
        with open('./data/infer/' + self.dataset + '/mRNA.fa') as f:
            for line in f:
                if line.startswith('>'):
                    name = line.strip().replace('>','')
                else:
                    seq = line.strip()
                    self.mRNA_ref[seq] = name

    def __del__(self):
        if getattr(self, "contact_store", None) is not None:
            self.contact_store.close()

    def _get_contact_key(self, row) -> str:
        if self.contact_key_col and self.contact_key_col in row:
            return row[self.contact_key_col]
        return f"{row['siRNA']}x{row['mRNA']}"

    def _get_default_contact_map(self, siRNA_seq: str, mRNA_seq: str):
        return np.zeros((len(siRNA_seq), len(mRNA_seq)), dtype=np.float32)

    def _get_p_value(self, row, attrs: dict):
        if self.contact_p_column and self.contact_p_column in row:
            try:
                return float(row[self.contact_p_column])
            except Exception:
                return 0.0
        for key in ("p_value", "p", "pval"):
            if attrs is not None and key in attrs:
                try:
                    return float(attrs[key])
                except Exception:
                    continue
        return 0.0

    def _load_contact_map_and_p(self, row):
        contact_map = None
        attrs = {}
        if self.contact_store is not None:
            key = self._get_contact_key(row)
            contact_map, attrs = self.contact_store.get(key)
        siRNA_seq = row["siRNA"]
        mRNA_seq = row["mRNA"]
        if contact_map is None:
            contact_map = self._get_default_contact_map(siRNA_seq, mRNA_seq)
        contact_map = np.asarray(contact_map, dtype=np.float32)
        p_value = self._get_p_value(row, attrs)
        return contact_map, np.float32(p_value)
        
    def __len__(self):
        return len(self.df_index)
    def __getitem__(self, index):
        index = self.df_index[index]
        # siRNA
        siRNA_seq = self.df.iloc[index]['siRNA']
        siRNA_seq = [*siRNA_seq]
        siRNA_seq = sequence_OneHot(siRNA_seq)
        siRNA = np.expand_dims(siRNA_seq, axis=2).transpose([2, 1, 0])
        # mRNA
        mRNA_seq = self.df.iloc[index]['mRNA']
        mRNA_seq = [*mRNA_seq]
        mRNA_seq = sequence_OneHot(mRNA_seq)
        mRNA = np.expand_dims(mRNA_seq, axis=2).transpose([2, 1, 0])
        # siRNA RNA-FM
        siRNA_seq = self.df.iloc[index]['siRNA']
        siRNA_FM = np.load('./data/infer/' + self.dataset + '/siRNA/representations/'+ str(self.siRNA_ref[siRNA_seq]) + '.npy')
        # mRNA RNA-FM
        mRNA_seq = self.df.iloc[index]['mRNA']
        mRNA_FM = np.load('./data/infer/' + self.dataset + '/mRNA/representations/' + str(self.mRNA_ref[mRNA_seq])+'.npy') 
        td = self.df.iloc[index]['td']
        td = torch.tensor(td).to(torch.float32)
        contact_map, contact_p = self._load_contact_map_and_p(self.df.iloc[index])
        return  siRNA, mRNA, siRNA_FM, mRNA_FM, td, contact_map, contact_p
