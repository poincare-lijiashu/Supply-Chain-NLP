# -*- coding: utf-8 -*-
"""轻量测试：v2 数据格式与配置一致性（v1 已归档 data/raw_v1，测试跟随当前主线 v2）。"""
import os

import pytest

from config.config import Config

os.environ["DATA_VERSION"] = "v2"  # smoke 套件固定 v2 主线（v1 回归走 git 历史）


@pytest.fixture(scope="module")
def conf():
    return Config()


def test_class_list(conf):
    assert len(conf.class_list) == 15
    assert conf.class_list[0] == "文件证件"
    assert conf.class_list[13] == "违禁限寄"
    assert conf.class_list[-1] == "其他/拒识"


def test_data_files_format(conf):
    for path in [conf.train_path, conf.dev_path, conf.test_path]:
        assert os.path.exists(path), f"先运行 data.generate_v2 生成 {path}"
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                parts = line.rstrip("\n").rsplit("\t", 1)
                assert len(parts) == 2, f"{path} 第{i + 1}行缺少制表符分隔"
                assert parts[1].isdigit() and 0 <= int(parts[1]) < 15
                if i > 200:
                    break


def test_config_paths(conf):
    assert os.path.exists(conf.class_path)
    assert os.path.exists(conf.stopwords_path)
    assert conf.confidence_threshold == 0.60


def test_distill_hparams(conf):
    assert conf.distill_T == 2.0 and conf.distill_alpha == 0.7
    assert conf.learning_rate == 5e-5 and conf.num_epochs == 4