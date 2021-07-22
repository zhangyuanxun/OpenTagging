import torch
import torch.nn as nn
import torch.nn.functional as F


class CRF(nn.Module):
    def __init__(self, num_tags, tag2id, device, batch_first=True):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.tag2id = tag2id
        self.device = device
        self.START_TAG = "[CLS]"
        self.STOP_TAG = "[SEP]"

        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.transitions.detach()[self.tag2id[self.START_TAG], :] = -10000
        self.transitions.detach()[:, self.tag2id[self.STOP_TAG]] = -10000
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))

        # initialize parameters
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

        self.start_transitions = self.start_transitions.to(device)
        self.end_transitions = self.start_transitions.to(device)

        self.transitions = self.transitions.to(device)
        self.batch_first = batch_first

    def _forward_alg(self, emissions, mask=None):
        seq_length = emissions.size(0)

        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        score += self.end_transitions

        return torch.logsumexp(score, dim=1)

    def _score_sentence(self, emissions, tags, mask):
        seq_length, batch_size = tags.shape
        mask = mask.type_as(emissions)

        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]
        return score

    def calculate_loss(self, emissions, tags, mask=None):
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)
        mask = mask.to(torch.bool)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # normalizer
        forward_score = self._forward_alg(emissions, mask)

        # correct path score
        gold_score = self._score_sentence(emissions, tags, mask)

        #  log likelihood = gold_score - normalizer, loss = -log likelihood
        score = forward_score - gold_score
        return score.mean()

    def viterbi_decode(self, emissions, mask=None):
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        mask = mask.to(torch.bool)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        seq_length, batch_size = mask.shape
        score = self.start_transitions + emissions[0]
        history = []

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list

    def obtain_labels(self, emissions, id2tag, mask=None):
        tags_list = []
        tag_seq_batch = self.viterbi_decode(emissions, mask)
        for tags in tag_seq_batch:
            tags_list.append([id2tag[tag] for tag in tags])

        return tags_list


if __name__ == "__main__":
    num_tags = 5
    tag_list = ['B-a', 'I-a', 'O', '[CLS]', '[SEP]']
    tag2id = {t: i for i, t in enumerate(tag_list)}
    id2tag = {i:t for i, t in enumerate(tag_list)}
    batch_size = 3
    seq_length = 7
    emissions = torch.randn(batch_size, seq_length, num_tags)
    tags = torch.tensor([[3, 1, 2, 2, 0, 1, 4], [3, 2, 0, 2, 1, 2, 4], [3, 0, 1, 2, 0, 1, 4]], dtype=torch.long)
    # tags = torch.tensor([[0, 1, 2, 0, 0, 1, 2]], dtype=torch.long)
    model = CRF(num_tags=num_tags, tag2id=tag2id, device='cpu')
    seq_len_tensor = torch.tensor([[seq_length], [seq_length], [seq_length]], dtype=torch.long)
    model.obtain_labels(emissions, id2tag, seq_length)
    #model.calculate_loss(emissions, tags)
    #model.viterbi_decode(emissions)