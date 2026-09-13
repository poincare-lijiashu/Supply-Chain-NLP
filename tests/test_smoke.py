# -*- coding: utf-8 -*-
"""轻量测试：数据格式与配置一致性（不依赖模型权重）。"""
import os

import pytest

from config.config import Config


@pytest.fixture(scope="module")
def conf():
    return Config()


def test_class_list(conf):
    assert len(conf.class_list) == 10
    assert conf.class_list[0] == "文件资料"
    assert conf.class_list[8] == "违禁品"
    assert "其他" in conf.class_list


def test_data_files_format(conf):
    for path in [conf.train_path, conf.dev_path, conf.test_path]:
        assert os.path.exists(path), f"先运行 data/generate_data.py 生成 {path}"
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                parts = line.rstrip("\n").rsplit("\t", 1)
                assert len(parts) == 2, f"{path} 第{i + 1}行缺少制表符分隔"
                assert parts[1].isdigit() and 0 <= int(parts[1]) < 10
                if i > 200:
                    break


def test_config_paths(conf):
    assert os.path.exists(conf.class_path)
    assert os.path.exists(conf.stopwords_path)
    assert conf.confidence_threshold == 0.60


def test_distill_hparams(conf):
    assert conf.distill_T == 2.0 and conf.distill_alpha == 0.7
    assert conf.learning_rate == 5e-5 and conf.num_epochs == 4
