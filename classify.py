"""Text classification using embedding similarity and SetFit fine-tuning."""

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

# 预定义的分类及其描述
CATEGORIES = {
    "政治": "政府政策、国家治理、政党、选举、外交关系、领导人、政治改革、立法、议会",
    "军事": "军队、武器装备、战争、国防、军事演习、导弹、航母、士兵、战斗机、军事冲突",
    "经济": "金融、贸易、股市、GDP、货币政策、银行、投资、通货膨胀、企业经营、市场",
    "科技": "人工智能、互联网、芯片、航天、新能源、软件、硬件、科学研究、创新技术",
}

# Few-shot 训练样本（每个类别 8-10 个样本）
TRAINING_SAMPLES = {
    "政治": [
        "国务院总理主持召开国务院常务会议，研究部署进一步优化营商环境的政策措施。",
        "全国人大常委会通过了新修订的《个人信息保护法》，将于明年正式实施。",
        "外交部发言人就中美关系问题回答记者提问，强调双方应加强对话合作。",
        "中央政治局召开会议，分析研究当前经济形势和经济工作。",
        "省委书记在党代会上作工作报告，总结过去五年发展成就。",
        "国家主席出席二十国集团领导人峰会并发表重要讲话。",
        "民政部发布新规，进一步规范社会组织登记管理工作。",
        "最高人民法院发布司法解释，明确网络侵权案件管辖规则。",
    ],
    "军事": [
        "我国自主研发的第三代战斗机成功完成首飞测试，采用先进隐身技术。",
        "海军航母编队在西太平洋开展远海训练，提升远洋作战能力。",
        "陆军某部举行实弹演习，检验部队快速反应和协同作战能力。",
        "国防部宣布将在南海举行例行军事演习，维护国家主权安全。",
        "空军新型预警机列装部队，大幅提升空中预警探测能力。",
        "火箭军某旅进行导弹发射训练，锤炼部队实战化打击本领。",
        "武警部队开展反恐演练，模拟处置多种突发事件。",
        "军事专家分析俄乌冲突局势，解读现代战争新特点。",
    ],
    "经济": [
        "央行宣布下调存款准备金率0.5个百分点，释放长期资金约1万亿元。",
        "上证指数今日大涨3%，成交额突破万亿元创年内新高。",
        "国家统计局公布数据显示，三季度GDP同比增长4.9%。",
        "商务部表示将进一步扩大对外开放，吸引更多外资企业投资。",
        "多家银行下调房贷利率，首套房贷款利率降至历史新低。",
        "人民币汇率持续走强，兑美元汇率升破7.0关口。",
        "发改委发布通知，调整国内成品油价格，汽油每吨上调150元。",
        "阿里巴巴发布财报，季度营收超2000亿元，云计算业务增长显著。",
    ],
    "科技": [
        "OpenAI发布GPT-5模型，在多项基准测试中刷新纪录。",
        "华为发布新一代麒麟芯片，采用先进3纳米工艺制程。",
        "SpaceX成功发射星舰飞船，完成首次轨道级试飞任务。",
        "中国科学院团队在量子计算领域取得重大突破，实现百量子比特操控。",
        "比亚迪推出新款电动汽车，续航里程突破1000公里。",
        "谷歌DeepMind发布新一代AI系统，在科学推理能力上超越人类专家。",
        "我国5G基站总数突破300万个，5G用户规模全球第一。",
        "字节跳动开源大语言模型，参数规模达千亿级别。",
    ],
}


class EmbeddingClassifier:
    """方案1: 使用 Sentence Transformer embedding 相似度进行分类"""

    def __init__(self, model_name: str = EMBEDDING_MODEL, threshold: float = 0.5):
        print(f"[Embedding分类器] 加载模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.categories = CATEGORIES
        self._category_embeddings = None
        self._category_names = None

    def _get_category_embeddings(self):
        """懒加载类别 embeddings"""
        if self._category_embeddings is None:
            self._category_names = list(self.categories.keys())
            category_texts = [f"{name}: {desc}" for name, desc in self.categories.items()]
            self._category_embeddings = self.model.encode(category_texts)
        return self._category_names, self._category_embeddings

    def classify(self, text: str, top_k: int = 3) -> list[dict]:
        """
        对文本进行分类，返回 top_k 个类别及其得分
        """
        category_names, category_embeddings = self._get_category_embeddings()
        text_embedding = self.model.encode([text])[0]

        # 计算余弦相似度
        similarities = np.dot(category_embeddings, text_embedding) / (
            np.linalg.norm(category_embeddings, axis=1) * np.linalg.norm(text_embedding)
        )

        sorted_indices = np.argsort(similarities)[::-1]
        results = []
        for idx in sorted_indices[:top_k]:
            results.append({
                "category": category_names[idx],
                "score": round(float(similarities[idx]), 4)
            })
        return results

    def classify_multi(self, text: str, threshold: float = None) -> list[str]:
        """多标签分类，返回所有超过阈值的类别"""
        if threshold is None:
            threshold = self.threshold
        results = self.classify(text, top_k=len(self.categories))
        return [r["category"] for r in results if r["score"] >= threshold]


class SetFitClassifier:
    """方案4: 使用 SetFit 进行 few-shot 微调分类"""

    def __init__(self, model_name: str = EMBEDDING_MODEL, model_path: str = None):
        """
        初始化 SetFit 分类器

        Args:
            model_name: 基础模型名称
            model_path: 已训练模型的保存路径，如果存在则加载
        """
        self.model_name = model_name
        self.model_path = model_path or "models/setfit_classifier"
        self.model = None
        self.label_names = list(TRAINING_SAMPLES.keys())

    def train(self, samples: dict = None, num_epochs: int = 1):
        """
        训练分类器

        Args:
            samples: 训练样本，格式为 {"类别": ["样本1", "样本2", ...]}
            num_epochs: 训练轮数
        """
        try:
            from datasets import Dataset
            from setfit import SetFitModel, Trainer, TrainingArguments
        except ImportError:
            raise ImportError("请安装依赖: pip install setfit datasets")

        if samples is None:
            samples = TRAINING_SAMPLES

        # 构建训练数据
        texts = []
        labels = []
        self.label_names = list(samples.keys())

        for label_idx, (category, sample_texts) in enumerate(samples.items()):
            for text in sample_texts:
                texts.append(text)
                labels.append(label_idx)

        train_dataset = Dataset.from_dict({
            "text": texts,
            "label": labels
        })

        print(f"[SetFit分类器] 训练数据: {len(texts)} 条, {len(self.label_names)} 个类别")
        print(f"[SetFit分类器] 类别: {self.label_names}")

        # 初始化模型
        print(f"[SetFit分类器] 加载基础模型: {self.model_name}")
        self.model = SetFitModel.from_pretrained(self.model_name)

        # 训练参数
        args = TrainingArguments(
            batch_size=16,
            num_epochs=num_epochs,
            evaluation_strategy="no",
            save_strategy="no",
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
        )

        print("[SetFit分类器] 开始训练...")
        trainer.train()
        print("[SetFit分类器] 训练完成!")

        # 保存模型
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.model_path)
        # 保存标签映射
        import json
        with open(f"{self.model_path}/label_names.json", "w", encoding="utf-8") as f:
            json.dump(self.label_names, f, ensure_ascii=False)
        print(f"[SetFit分类器] 模型已保存到: {self.model_path}")

    def load(self):
        """加载已训练的模型"""
        try:
            from setfit import SetFitModel
        except ImportError:
            raise ImportError("请安装依赖: pip install setfit")

        import json

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"模型不存在: {self.model_path}，请先调用 train() 训练")

        print(f"[SetFit分类器] 加载模型: {self.model_path}")
        self.model = SetFitModel.from_pretrained(self.model_path)

        label_file = f"{self.model_path}/label_names.json"
        if Path(label_file).exists():
            with open(label_file, encoding="utf-8") as f:
                self.label_names = json.load(f)

    def classify(self, text: str) -> list[dict]:
        """
        对文本进行分类

        Returns:
            list of dict: [{"category": "政治", "score": 0.85}, ...]
        """
        if self.model is None:
            self.load()

        # SetFit 预测
        predictions = self.model.predict_proba([text])[0]

        results = []
        for idx, score in enumerate(predictions):
            results.append({
                "category": self.label_names[idx],
                "score": round(float(score), 4)
            })

        # 按分数降序排列
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def classify_multi(self, text: str, threshold: float = 0.3) -> list[str]:
        """多标签分类，返回所有超过阈值的类别"""
        results = self.classify(text)
        return [r["category"] for r in results if r["score"] >= threshold]


def read_file(file_path: str) -> str:
    """读取文件内容"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    return path.read_text(encoding="utf-8")


def classify_file(file_path: str, method: str = "embedding", **kwargs) -> dict:
    """
    对文件进行分类

    Args:
        file_path: 文件路径
        method: 分类方法，"embedding" 或 "setfit"
    """
    content = read_file(file_path)

    if method == "embedding":
        classifier = EmbeddingClassifier(**kwargs)
        results = classifier.classify(content, top_k=4)
        return {"file": file_path, "results": results, "method": "embedding"}
    elif method == "setfit":
        classifier = SetFitClassifier(**kwargs)
        results = classifier.classify(content)
        return {"file": file_path, "results": results, "method": "setfit"}
    else:
        raise ValueError(f"未知的分类方法: {method}")


if __name__ == "__main__":
    # 测试文本
    test_texts = [
        "国务院总理今日主持召开国务院常务会议，研究部署进一步优化营商环境的政策措施，深化放管服改革。",
        "我国自主研发的第三代战斗机成功完成首飞测试，采用了先进的隐身技术，国防部表示将大幅提升空中作战能力。",
        "央行今日宣布下调存款准备金率0.5个百分点，释放长期资金约1万亿元，A股市场全线上涨。",
        "OpenAI发布了最新的GPT-5模型，百度、阿里也相继发布了各自的大模型新版本，AI竞争日趋激烈。",
        "外交部就南海军事演习问题回应记者提问，强调中国有权在自己领土进行正常军事训练。",  # 政治+军事
    ]

    print("=" * 70)
    print("方案1: Embedding 相似度分类")
    print("=" * 70)

    emb_classifier = EmbeddingClassifier(threshold=0.4)

    for i, text in enumerate(test_texts, 1):
        print(f"\n[测试 {i}] {text[:50]}...")
        results = emb_classifier.classify(text, top_k=4)
        for r in results:
            bar = "█" * int(r["score"] * 20)
            print(f"  {r['category']}: {r['score']:.4f} {bar}")
        multi = emb_classifier.classify_multi(text, threshold=0.5)
        print(f"  → 多标签结果(阈值0.5): {multi}")

    print("\n" + "=" * 70)
    print("方案4: SetFit 微调分类器")
    print("=" * 70)

    try:
        setfit_classifier = SetFitClassifier()

        # 检查是否已有训练好的模型
        if not Path(setfit_classifier.model_path).exists():
            print("\n首次运行，开始训练模型...")
            setfit_classifier.train(num_epochs=1)
        else:
            print("\n加载已训练的模型...")
            setfit_classifier.load()

        for i, text in enumerate(test_texts, 1):
            print(f"\n[测试 {i}] {text[:50]}...")
            results = setfit_classifier.classify(text)
            for r in results:
                bar = "█" * int(r["score"] * 20)
                print(f"  {r['category']}: {r['score']:.4f} {bar}")
            multi = setfit_classifier.classify_multi(text, threshold=0.3)
            print(f"  → 多标签结果(阈值0.3): {multi}")

    except ImportError as e:
        print(f"\n{e}")
        print("运行以下命令安装依赖:")
        print("  pip install setfit datasets")
