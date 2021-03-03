from models import MPAD
import torch
from torch import optim
from tqdm import tqdm
import numpy as np

class Learner:

	def __init__(self, experiment_name, device):

		self.model = None
		self.optimizer = None
		self.scheduler = None
		self.device = device
		self.writer = None
		self.train_step = 0

	def init_model(self,
				   model_type='mpad',
				   lr=0.1,
				   **kwargs
				   ):
		if model_type.lower() == 'mpad':
			self.model = MPAD(**kwargs)
		else:
			raise AssertionError("Currently only MPAD is supported as model")

		self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
		self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)


	def train_epoch(self, dataloader, eval_every):

		self.model.train()
		total_iters = -1

		with tqdm(initial=0, total=eval_every) as pbar_train:
			for batch_ix, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
				total_iters += 1

				batch = (t.to(self.device) for t in batch)
				A, nodes, y = batch

				preds = self.model(nodes, A)

				loss = self.criterion(preds, y)

				self.optimizer.zero_grad()
				loss.backward()

				# grad norm clipping?
				self.optimizer.step()
				self.scheduler.step()

				pbar_train.update(1)
				pbar_train.set_description(
					"finetuning phase {} -> loss: {}".format(
						total_iters + 1, loss.item()
					)
				)

				if (total_iters + 1) % eval_every == 0:
					# Stop training
					break



	def evaluate(self, dataloader):

		self.model.eval()
		y_pred = []
		y_true = []
		running_loss = 0

		######################################
		# Infer the model on the dataset
		######################################
		with tqdm(initial=0, total=len(dataloader)) as pbar_eval:
			with torch.no_grad():
				for batch_idx, batch in enumerate(dataloader):
					batch = (t.to(self.device) for t in batch)
					A, nodes, y = batch

					preds = self.model(nodes, A)
					loss = self.criterion(preds, y)
					running_loss += loss.item()
					# store predictions and targets
					y_pred.extend(list(preds.cpu().detach().numpy()))
					y_true.extend(list(np.round(y.cpu().detach().numpy())))

					pbar_eval.update(1)
					pbar_eval.set_description(
						"Eval step {} -> loss: {}".format(
							batch_idx + 1, loss.item()
						)
					)
		######################################
		# Compute metrics
		######################################


