import torch
import torch.nn.functional as F
from einops import einsum
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import nn


class MultiHeadedAttention(nn.Module):
    """Implements a multi-headed attention mechanism.

    Parameters:
        d_model: Dimensionality of the model.
        num_heads: Number of attention heads.
        causal: If True, use causality to mask future values.
        rotary_emb: Instance of RotaryEmbedding for positional encoding.
        num_embeddings: The number of embeddings (unused in this class).

    Attributes:
        num_heads: Stored number of heads.
        causal: Stores whether causality is used.
        rotary_emb: Stores the RotaryEmbedding instance.
        q_projection: Linear projection for queries.
        k_projection: Linear projection for keys.
        v_projection: Linear projection for values.
        o_projection: Linear projection for output.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        causal: bool = False,
        rotary_emb: RotaryEmbedding = None,
        num_embeddings: int = 64,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.causal = causal
        self.rotary_emb = rotary_emb
        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)
        self.o_projection = nn.Linear(d_model, d_model)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for the MultiHeadedAttention layer.

        Args:
            q: Query tensor.
            k: Key tensor.
            v: Value tensor.

        Returns:
            Output tensor after computing multi-headed attention.
        """
        batch_size, seq_length, d_model = q.shape

        # Project and reshape query, key, and value tensors
        q = rearrange(
            self.q_projection(q),
            "b l (head d) -> b head l d",
            head=self.num_heads,
        )
        k = rearrange(
            self.k_projection(k),
            "b l (head d) -> b head l d",
            head=self.num_heads,
        )
        v = rearrange(
            self.v_projection(v),
            "b l (head d) -> b head l d",
            head=self.num_heads,
        )

        # Apply rotary embeddings
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # Compute scaled dot-product attention scores
        attention = (
            einsum(q, k, "b h l d, b h t d -> b h l t")
            / (d_model // self.num_heads) ** 0.5
        )

        # Mask future values if causal attention is required
        if self.causal:
            mask = (
                torch.ones((seq_length, seq_length))
                .tril()
                .bool()
                .type_as(attention)
            )
            attention.masked_fill_(mask.logical_not(), float("-inf"))

        # Softmax over the last dimension to get the attention probabilities
        attention_probs = F.softmax(attention, dim=-1)

        # Compute weighted sum of values based on attention probabilities
        attention = einsum(attention_probs, v, "b h l t, b h t d -> b h l d")

        # Merge the heads and project to output size
        attention = rearrange(attention, "b h l d -> b l (h d)")
        out = self.o_projection(attention)

        return out


class EncoderBlock(nn.Module):
    """Implements an encoder block with multi-headed attention and feedforward
    neural network layers.

    Parameters:
        d_model: Dimensionality of the model.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        rotary_emb: Instance of RotaryEmbedding for positional encoding.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        rotary_emb: RotaryEmbedding = None,
    ) -> None:
        super().__init__()
        self.multi_headed_attention = MultiHeadedAttention(
            d_model,
            num_heads,
            causal=False,
            rotary_emb=rotary_emb,
        )
        self.feedforward_layer = nn.Linear(d_model, d_model)
        self.activation_fn = nn.GELU()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the EncoderBlock.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the encoder block.
        """
        x = self.layer_norm_1(
            self.dropout(self.multi_headed_attention(x, x, x) + x)
        )
        x = self.layer_norm_2(
            self.dropout(self.activation_fn(self.feedforward_layer(x)) + x)
        )
        return x


class Encoder(nn.Module):
    """Implements an encoder consisting of a stack of EncoderBlocks.

    Parameters:
        d_model: Dimensionality of the model.
        num_heads: Number of attention heads.
        num_layers: Number of stacked EncoderBlocks.
        dropout: Dropout rate.
        rotary_emb: Instance of RotaryEmbedding for positional encoding.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
        rotary_emb: RotaryEmbedding = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    rotary_emb=rotary_emb,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all encoder blocks.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the encoder stack.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderBlock(nn.Module):
    """Implements a decoder block with masked multi-headed attention and multi-
    headed attention layers, followed by a position-wise feedforward network.

    Parameters:
        d_model: Dimensionality of the model.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        rotary_emb: A custom positional embedding layer.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        rotary_emb: RotaryEmbedding = None,
    ) -> None:
        super().__init__()
        self.masked_multi_headed_attention = MultiHeadedAttention(
            d_model,
            num_heads,
            causal=True,
            rotary_emb=rotary_emb,
        )
        self.multi_headed_attention = MultiHeadedAttention(
            d_model,
            num_heads,
            causal=False,
            rotary_emb=rotary_emb,
        )
        self.feedforward_layer = nn.Linear(d_model, d_model)
        self.activation_fn = nn.GELU()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.layer_norm_3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for the DecoderBlock.

        Args:
            q: Query tensor.
            k: Key tensor.
            v: Value tensor.

        Returns:
            Output tensor after passing through the decoder block.
        """
        x = self.layer_norm_1(
            self.dropout(self.masked_multi_headed_attention(q, q, q) + q)
        )
        x = self.layer_norm_2(
            self.dropout(self.multi_headed_attention(x, k, v) + x)
        )
        x = self.layer_norm_3(
            self.dropout(self.activation_fn(self.feedforward_layer(x)) + x)
        )
        return x


class Decoder(nn.Module):
    """Implements a decoder consisting of a stack of DecoderBlocks.

    Parameters:
        d_model: Dimensionality of the model.
        num_heads: Number of attention heads.
        num_layers: Number of stacked DecoderBlocks.
        dropout: Dropout rate.
        rotary_emb: A custom positional embedding layer.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
        rotary_emb: RotaryEmbedding = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    rotary_emb=rotary_emb,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through all decoder blocks.

        Args:
            q: Query tensor.
            k: Key tensor.
            v: Value tensor.

        Returns:
            Output tensor after passing through the decoder stack.
        """
        for layer in self.layers:
            q = layer(q, k, v)
        return q


class TransformerModel(nn.Module):
    """Implements a Transformer model consisting of an encoder-decoder
    architecture.

    Parameters:
        d_model: Dimensionality of the model.
        num_heads: Number of attention heads.
        num_layers: Number of encoder/decoder layers.
        dropout: Dropout rate.
        rotary_embedding_dim: Dimensionality of rotary embeddings.
        enc_vocab_size: Vocabulary size for the encoder.
        dec_vocab_size: Vocabulary size for the decoder.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
        rotary_embedding_dim: int = 32,
        enc_vocab_size: int = None,
        dec_vocab_size: int = None,
    ):
        super().__init__()
        self.encoder_embeddings = nn.Embedding(enc_vocab_size, d_model)
        self.decoder_embeddings = nn.Embedding(dec_vocab_size, d_model)
        rotary_emb = RotaryEmbedding(dim=rotary_embedding_dim)

        self.encoder = Encoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            rotary_emb=rotary_emb,
        )
        self.decoder = Decoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            rotary_emb=rotary_emb,
        )
        self.lm_head = nn.Linear(d_model, dec_vocab_size)

    def forward(
        self, input_ids: torch.Tensor, decoder_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for the TransformerModel.

        Args:
            input_ids: Token IDs for the inputs.
            decoder_ids: Token IDs for the output predictions during training.

        Returns:
            Logits representing the prediction scores for each token in the
            output vocabulary.
        """
        encoder_embed = self.encoder_embeddings(input_ids)
        decoder_embed = self.decoder_embeddings(decoder_ids)

        encoder_embed = self.encoder(encoder_embed)

        decoder_embed = self.decoder(
            decoder_embed, encoder_embed, encoder_embed
        )
        logits = self.lm_head(decoder_embed)

        return logits
