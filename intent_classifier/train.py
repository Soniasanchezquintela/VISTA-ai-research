import csv
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, List, Optional, Sequence

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split

from intent_classifier.classifier import (
    INTENT_TO_ID,
    INTENTS,
    IntentClassifierModel,
    get_default_device,
)
from intent_classifier.schema import Intent


CONFIG_PATH = Path(__file__).resolve().parent / "train.yaml"


@dataclass(frozen=True)
class TrainingConfig:
    data: Path
    output: Path
    metrics_figure: Path
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_length: int
    hidden_size: int
    dropout: float
    validation_split: float
    seed: int
    device: Optional[str]
    freeze_encoder: bool


@dataclass(frozen=True)
class IntentExample:
    text: str
    intent: Intent
    target: Optional[str]


class IntentDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[IntentExample],
        tokenizer,
        max_length: int,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        encoding = self.tokenizer(
            example.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(INTENT_TO_ID[example.intent], dtype=torch.long),
        }


def load_examples(csv_path: Path) -> List[IntentExample]:
    examples: List[IntentExample] = []

    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {"text", "intent", "target"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path} is missing required column(s): {missing}")

        for line_number, row in enumerate(reader, start=2):
            text = (row["text"] or "").strip()
            intent_value = (row["intent"] or "").strip()
            target = (row["target"] or "").strip() or None

            if not text:
                raise ValueError(f"{csv_path}:{line_number} has an empty text value")

            try:
                intent = Intent(intent_value)
            except ValueError as exc:
                raise ValueError(
                    f"{csv_path}:{line_number} has unknown intent label: {intent_value!r}"
                ) from exc

            examples.append(IntentExample(text=text, intent=intent, target=target))

    if not examples:
        raise ValueError(f"{csv_path} does not contain any training examples")

    return examples


def split_dataset(dataset: Dataset, validation_split: float, seed: int) -> tuple[Dataset, Optional[Dataset]]:
    if validation_split <= 0.0:
        return dataset, None

    validation_size = max(1, round(len(dataset) * validation_split))
    train_size = len(dataset) - validation_size
    if train_size <= 0:
        raise ValueError("validation_split leaves no examples for training")

    generator = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )
    return train_dataset, validation_dataset


def run_epoch(
    model: IntentClassifierModel,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return average_loss, accuracy


def load_config(config_path: Path = CONFIG_PATH) -> TrainingConfig:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "Training configuration requires the 'PyYAML' package. "
            "Install the project requirements before running this module."
        ) from exc

    with config_path.open(encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    if not isinstance(raw_config, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")

    valid_keys = {field.name for field in fields(TrainingConfig)}
    unknown_keys = set(raw_config) - valid_keys
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"{config_path} contains unknown key(s): {unknown}")

    missing_keys = valid_keys - set(raw_config)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"{config_path} is missing required key(s): {missing}")

    config_values: dict[str, Any] = dict(raw_config)
    config_dir = config_path.parent
    for path_key in ("data", "output", "metrics_figure"):
        path = Path(config_values[path_key])
        if not path.is_absolute():
            path = config_dir / path
        config_values[path_key] = path

    return TrainingConfig(**config_values)


def save_metrics_figure(
    train_losses: Sequence[float],
    train_accuracies: Sequence[float],
    validation_losses: Sequence[float],
    validation_accuracies: Sequence[float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))

    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_losses) + 1))
    train_accuracies_percent = [accuracy * 100.0 for accuracy in train_accuracies]
    validation_accuracies_percent = [
        accuracy * 100.0 for accuracy in validation_accuracies
    ]

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epochs, train_losses, label="train")
    if validation_losses:
        plt.plot(epochs, validation_losses, label="test")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy [%]")
    plt.plot(epochs, train_accuracies_percent, label="train")
    if validation_accuracies_percent:
        plt.plot(epochs, validation_accuracies_percent, label="test")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train(config: TrainingConfig) -> Path:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Training requires the 'transformers' package. "
            "Install the project requirements before running this module."
        ) from exc

    torch.manual_seed(config.seed)

    data_path = config.data
    output_path = config.output
    examples = load_examples(data_path)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    dataset = IntentDataset(examples, tokenizer=tokenizer, max_length=config.max_length)
    train_dataset, validation_dataset = split_dataset(
        dataset,
        config.validation_split,
        config.seed,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size)

    device = torch.device(config.device) if config.device is not None else get_default_device()
    model = IntentClassifierModel(
        model_name=config.model_name,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        freeze_encoder=config.freeze_encoder,
    ).to(device)

    optimizer = AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    print(f"Loaded {len(examples)} examples from {data_path}")
    print(f"Training on {device} for {config.epochs} epoch(s)")

    best_validation_loss: Optional[float] = None
    best_state_dict = None
    train_losses: List[float] = []
    train_accuracies: List[float] = []
    validation_losses: List[float] = []
    validation_accuracies: List[float] = []

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        message = (
            f"Epoch {epoch}/{config.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.3f}"
        )

        if validation_loader is not None:
            validation_loss, validation_accuracy = run_epoch(
                model=model,
                dataloader=validation_loader,
                loss_fn=loss_fn,
                device=device,
            )
            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_accuracy)
            message += (
                f" val_loss={validation_loss:.4f} "
                f"val_acc={validation_accuracy:.3f}"
            )

            if best_validation_loss is None or validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state_dict = {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                }

        print(message)

    save_metrics_figure(
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        validation_losses=validation_losses,
        validation_accuracies=validation_accuracies,
        output_path=config.metrics_figure,
    )
    print(f"Saved metrics figure to {config.metrics_figure}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if best_state_dict is not None:
        model_state_dict = best_state_dict
    else:
        model_state_dict = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }

    torch.save(
        {
            "model_state_dict": model_state_dict,
            "model_name": config.model_name,
            "max_length": config.max_length,
            "hidden_size": config.hidden_size,
            "dropout": config.dropout,
            "intents": [intent.value for intent in INTENTS],
        },
        output_path,
    )
    print(f"Saved checkpoint to {output_path}")

    return output_path


def main() -> None:
    train(load_config())


if __name__ == "__main__":
    main()
