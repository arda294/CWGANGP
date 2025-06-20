import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DataColumnInfo, TabularDataset, DataTransformer, DataSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from safetensors.torch import save_file, safe_open
import pickle

class CWGANGPGenerator(nn.Module):
    def __init__(self, input_dim: int, data_column_infos: dict, hidden_dim: int = 256, device=None):
        super(CWGANGPGenerator, self).__init__()
        self.input_dim = input_dim
        self.data_column_infos = data_column_infos
        self.hidden_dim = hidden_dim
        self.device = torch.device(device if torch.cuda.is_available() and device != None else "cpu")
        self.discrete_cols: list[DataColumnInfo] = [col_info for _, col_info in self.data_column_infos.items() if col_info.is_discrete == True]
        self.numerical_cols: list[DataColumnInfo] = [col_info for _, col_info in self.data_column_infos.items() if col_info.is_discrete == False]
        self.debug = False

        print(self.discrete_cols)

        # Get conditional and numerical dims
        self.cond_dim = sum(len(discrete_col.category_counts) for discrete_col in self.discrete_cols)
        self.numerical_dim = sum(numerical_col.num_clusters + 1 for numerical_col in self.numerical_cols)

        print(f"cond_dim={self.cond_dim}")
        print(f"numerical_dim={self.numerical_dim}")

        # Fully connected layers
        # |cond| + |z| -> 256
        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dim + self.cond_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        ) 
        # |cond| + |z| + 256 -> 256
        self.fc2 = nn.Sequential(
            nn.Linear(self.input_dim + self.cond_dim + self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        ) 

        # Output layers
        self.fc_numeric_normalized = nn.ModuleList()
        self.fc_numeric_component = nn.ModuleList()
        self.fc_cond = nn.ModuleList()

        for numerical_col in self.numerical_cols:
            # Normalized column use tanh
            # |cond| + |z| + 2 * 256 -> 1
            self.fc_numeric_normalized.append(nn.Sequential(
                    nn.Linear(self.input_dim + self.cond_dim + 2*self.hidden_dim, 1),
                    nn.Tanh()
                )
            ) 
            # Mode indicator use gumbel softmax (added in forward pass)
            # |cond| + |z| + 2 * 256 -> total modes
            self.fc_numeric_component.append(nn.Linear(self.input_dim + self.cond_dim + 2*self.hidden_dim, numerical_col.num_clusters)) 
            
        for discrete_col in self.discrete_cols:        # Discrete value use gumbel softmax (added in forward pass)
            # |cond| + |z| + 2 * 256 -> total discrete dims
            self.fc_cond.append(nn.Linear(self.input_dim + self.cond_dim + 2*self.hidden_dim, len(discrete_col.category_counts)))
        
        print(f"using {self.device}")
        self.to(self.device)

    def forward(self, z: torch.Tensor, cond: torch.Tensor):
        h0 = torch.cat((z, cond), dim=1)

        h0_out = self.fc1(h0)
        h1 = torch.cat((h0, h0_out), dim=1)

        h2_out = self.fc2(h1)
        h2 = torch.cat((h1, h2_out), dim=1)

        numerical_outputs = []
        for fc_normalized, fc_component in zip(self.fc_numeric_normalized, self.fc_numeric_component):
            a_i = fc_normalized(h2)
            b_i = F.gumbel_softmax(fc_component(h2), tau=0.2, hard=False)
            numerical_outputs.append(a_i)
            numerical_outputs.append(b_i)
        
        numerical_outputs = torch.cat(numerical_outputs, dim=1)

        discrete_outputs = []
        for fc_discrete in self.fc_cond:
            d_i = F.gumbel_softmax(fc_discrete(h2), tau=0.2, hard=False)
            discrete_outputs.append(d_i)

        discrete_outputs = torch.cat(discrete_outputs, dim=1)

        output = torch.cat((numerical_outputs, discrete_outputs), dim=1)
        return output
    
class CWGANGPCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, device=None):
        super(CWGANGPCritic, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device if torch.cuda.is_available() and device != None else "cpu")

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )

        self.output = nn.Linear(self.hidden_dim, 1)

        self.to(device=self.device)

    def forward(self, inputs: torch.Tensor, cond: torch.Tensor):
        h0 = torch.cat((inputs, cond), dim=1)

        h1 = self.fc1(h0)
        h2 = self.fc2(h1)
        
        output = self.output(h2)
        return output
    
class CWGANGP():
    def __init__(
            self,
            generator_input_dim: int = 128,
            generator_hidden_dim: int = 256,
            critic_hidden_dim: int = 256,
            generator_lr: float = 1e-4,
            generator_decay: float = 1e-6,
            critic_steps: int = 1,
            critic_lr: float = 1e-4,
            critic_decay: float = 1e-6,
            use_column_pmf: bool = False,
            epochs: int = 300,
            batch_size: int = 500,
            transformer_fit_n_jobs = None,
            transformer_transform_n_jobs = None,
            device=None
        ):

        self._generator_input_dim = generator_input_dim
        self._critic_input_dim = None
        self._generator_hidden_dim = generator_hidden_dim
        self._critic_hidden_dim = critic_hidden_dim
        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._critic_steps = critic_steps
        self._critic_lr = critic_lr
        self._critic_decay = critic_decay
        self._use_column_pmf = use_column_pmf
        self._epochs = epochs
        self._batch_size = batch_size

        self._transformer_fit_n_jobs = transformer_fit_n_jobs
        self._transformer_transform_n_jobs = transformer_transform_n_jobs

        self._device = torch.device(device if torch.cuda.is_available() and device != None else "cpu")

        self._generator = None
        self._critic = None
        self._optimizer_generator = None
        self._optimizer_critic = None

        self._numerical_dim = None
        self._cond_dim = None

        self._transformer = None
        self._transformed_data = None
        self._dataset = None
        self._sampler = None

        self._last_epoch = 0
        self._losses = []

    def fit(self, raw_data: pd.DataFrame, discrete_columns: list[str], epochs: (int | None) = None):
        # Override epochs
        self._epochs = epochs if epochs else self._epochs

        # Preprocess raw data
        if self._transformer is None:
            self._transformer = DataTransformer(fit_n_jobs=self._transformer_fit_n_jobs, transform_n_jobs=self._transformer_transform_n_jobs)
            self._transformer.fit(raw_data=raw_data, discrete_columns=discrete_columns)
            self._transformed_data = self._transformer.transform(raw_data=raw_data)
            self._transformed_columns = self._transformed_data.columns
        else:
            self._transformed_data = self._transformer.transform(raw_data=raw_data)
        
        # Get conditional and numerical dims
        if self._cond_dim is None or self._numerical_dim is None:
            discrete_cols: list[DataColumnInfo] = [col_info for _, col_info in self._transformer.data_column_infos.items() if col_info.is_discrete == True]
            numerical_cols: list[DataColumnInfo] = [col_info for _, col_info in self._transformer.data_column_infos.items() if col_info.is_discrete == False]
            
            self._cond_dim = sum(len(discrete_col.category_counts) for discrete_col in discrete_cols)
            self._numerical_dim = sum(numerical_col.num_clusters + 1 for numerical_col in numerical_cols)
            self._critic_input_dim = len(self._transformed_data.columns) + self._cond_dim

        # Prepare sampler
        self._dataset = TabularDataset(dataset=self._transformed_data, numerical_dim=self._numerical_dim, discrete_column_counts=self._transformer.category_counts())
        self._sampler = DataSampler(dataset=self._dataset, batch_size=self._batch_size, use_column_pmf=self._use_column_pmf)

        # Initialize generator
        if self._generator is None:
            self._generator = CWGANGPGenerator(
                input_dim=self._generator_input_dim,
                hidden_dim=self._generator_hidden_dim,
                data_column_infos=self._transformer.data_column_infos,
                device=self._device
            )

        # Initialize critic    
        if self._critic is None:
            self._critic = CWGANGPCritic(
                input_dim=self._critic_input_dim,
                hidden_dim=self._critic_hidden_dim,
                device=self._device
            )

        # Initialize generator optimizer
        if self._optimizer_generator is None:
            self._optimizer_generator = torch.optim.Adam(
                params=self._generator.parameters(),
                lr=self._generator_lr,
                betas=(0.5, 0.9),
                eps=1e-8,
                weight_decay=self._generator_decay
            )

        # Initialize critic optimizer
        if self._optimizer_critic is None:
            print('test')
            self._optimizer_critic = torch.optim.Adam(
                params=self._critic.parameters(),
                lr=self._critic_lr,
                betas=(0.5, 0.9),
                eps=1e-8,
                weight_decay=self._critic_decay     
            )

        self._last_epoch = len(self._losses)


        if self._last_epoch > 0:
            print(f"Resuming training from epoch:{self._last_epoch}")
            
            print(f"generator_lr: {self._optimizer_generator.param_groups[0]['lr']}")
            print(f"critic_lr: {self._optimizer_critic.param_groups[0]['lr']}")
        
            last_loss = self._losses[-1]
            print(f"last recorded g_loss: {last_loss['generator_loss']:.4f}, c_loss: {last_loss['critic_loss']:.4f}")

        # Do training loop
        self._train_loop()

    def _train_loop(self):
        steps_per_epoch = len(self._dataset) // self._batch_size

        generator_loss = None
        critic_loss = None
        
        tqdm._instances.clear()
        pbar = tqdm(total=self._epochs, ncols=100, leave=True)

        for epoch in range(self._epochs):
            for step in range(steps_per_epoch):
                for _ in range(self._critic_steps):
                    col_idx, _, cond_vector, batch_indices = self._sampler.get_samples()
                        
                    # Generate random noise
                    z = torch.randn(self._batch_size, self._generator_input_dim, device=self._device)

                    cond_vector = torch.from_numpy(cond_vector).to(device=self._device)

                    fake_datas = self._generator(z, cond_vector)
                    real_datas = self._dataset[batch_indices].to(device=self._device)

                    y_fake = self._critic(fake_datas, cond_vector)
                    y_real = self._critic(real_datas, cond_vector)

                    # Calculate critic loss (wgan loss with gradient penalty)
                    critic_loss = self._critic_loss(real_datas=real_datas, fake_datas=fake_datas, y_real=y_real, y_fake=y_fake, cond_vector=cond_vector)

                    # Backprop
                    self._optimizer_critic.zero_grad(set_to_none=False)
                    critic_loss.backward()
                    self._optimizer_critic.step()

                # Generator train step
                col_idx, mask_vector, cond_vector, batch_indices = self._sampler.get_samples()

                # Generate random noise
                z = torch.randn(self._batch_size, self._generator_input_dim, device=self._device)
                cond_vector = torch.from_numpy(cond_vector).to(device=self._device)
                mask_vector = torch.from_numpy(mask_vector).to(device=self._device)

                fake_datas = self._generator(z, cond_vector)

                y_fake = self._critic(fake_datas, cond_vector)

                generator_loss = self._generator_loss(fake_datas=fake_datas, y_fake=y_fake, cond_vector=cond_vector, mask_vectors=mask_vector, col_idx=col_idx)

                self._optimizer_generator.zero_grad(set_to_none=False)
                generator_loss.backward()
                self._optimizer_generator.step()
                
                pbar.set_description(desc=f"steps {step+1:{' '}>3}/{steps_per_epoch}, c_loss: {critic_loss:.2f}, g_loss {generator_loss:.2f}")

            pbar.update()

            self._losses.append({
                'epoch': self._last_epoch + epoch + 1,
                'critic_loss': critic_loss.detach().cpu().item(),
                'generator_loss': generator_loss.detach().cpu().item()
            })

        pbar.close()  
        # End training

    def save_model(self, path: str):
        state_dict = {}

        # Generator state dicts
        for name, tensors in self._generator.state_dict().items():
            state_dict[f"generator.{name}"] = tensors

        # Critic state dicts
        for name, tensors in self._critic.state_dict().items():
            state_dict[f"critic.{name}"] = tensors
        
        metadata = {
            "generator_input_dim": str(self._generator_input_dim),
            "generator_hidden_dim": str(self._generator_hidden_dim),
            "critic_hidden_dim": str(self._critic_hidden_dim),
            "critic_input_dim": str(self._critic_input_dim),
            "generator_lr": str(self._generator_lr),
            "generator_decay": str(self._generator_decay),
            "use_column_pmf": str(self._use_column_pmf),
            "critic_steps": str(self._critic_steps),
            "critic_lr": str(self._critic_lr),
            "critic_decay": str(self._critic_decay),
            "batch_size": str(self._batch_size),
            "epochs": str(self._epochs),
            "numerical_dim": str(self._numerical_dim),
            "cond_dim": str(self._cond_dim),
            "device": str(self._device),
        }

        save_file(tensors=state_dict, filename=path, metadata=metadata)

        additional_data = {
            "transformed_columns": self._transformed_columns,
            "transformer_fit_n_jobs": self._transformer.fit_n_jobs,
            "transformer_transform_n_jobs": self._transformer.transform_n_jobs,
            "transformer_discrete_transformer": self._transformer.discrete_transformer,
            "transformer_data_column_infos": self._transformer.data_column_infos,
            "transformer_raw_data_column_order": self._transformer.raw_data_column_order,
            "losses": self._losses,
            "optimizer_generator_state": self._optimizer_generator.state_dict(),
            "optimizer_critic_state": self._optimizer_critic.state_dict()
        }

        additional_path = path.replace('.safetensors', '.pkl')
        with open(additional_path, 'wb') as f:
            pickle.dump(additional_data, f)

        print(f"Model saved to {path}")

    @staticmethod
    def load_model(path: str, device: str | None = None):
        state_dict = {}
        metadata = None

        use_device = torch.device(device if torch.cuda.is_available() and device != None else "cpu")

        with safe_open(filename=path, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key).clone()

            metadata = f.metadata()

        model = CWGANGP(
            generator_input_dim=int(metadata["generator_input_dim"]),
            generator_hidden_dim=int(metadata["generator_hidden_dim"]),
            critic_hidden_dim=int(metadata["critic_hidden_dim"]),
            generator_lr=float(metadata["generator_lr"]),
            generator_decay=float(metadata["generator_decay"]),
            critic_steps=int(metadata["critic_steps"]),
            critic_lr=float(metadata["critic_lr"]),
            critic_decay=float(metadata["critic_decay"]),
            use_column_pmf=bool(metadata["use_column_pmf"]),
            batch_size=int(metadata["batch_size"]),
            epochs=int(metadata["epochs"]),
            device=use_device
        )

        model._numerical_dim = int(metadata["numerical_dim"])
        model._cond_dim = int(metadata["cond_dim"])
        model._critic_input_dim = int(metadata["critic_input_dim"])

        additional_path = path.replace('.safetensors', '.pkl')

        try:
            with open(additional_path, 'rb') as f:
                additional_data = pickle.load(f)
        except FileNotFoundError:
            print("Cant find model pickle file")
            return

        model._transformer = DataTransformer.load_state(
            fit_n_jobs=additional_data["transformer_fit_n_jobs"],
            transform_n_jobs=additional_data["transformer_transform_n_jobs"],
            discrete_transformer=additional_data["transformer_discrete_transformer"],
            data_column_infos=additional_data["transformer_data_column_infos"],
            raw_data_column_order=additional_data["transformer_raw_data_column_order"]
        )

        model._transformed_columns = additional_data["transformed_columns"]

        model._generator = CWGANGPGenerator(
            input_dim=model._generator_input_dim,
            hidden_dim=model._generator_hidden_dim,
            data_column_infos=model._transformer.data_column_infos,
            device=model._device
        )
        
        model._critic = CWGANGPCritic(
            input_dim=model._critic_input_dim,
            hidden_dim=model._critic_hidden_dim,
            device=model._device
        )

        generator_state = {k.replace("generator.", ""): v for k, v in state_dict.items() if k.startswith("generator.")}
        critic_state = {k.replace("critic.", ""): v for k, v in state_dict.items() if k.startswith("critic.")}

        model._generator.load_state_dict(generator_state)
        model._critic.load_state_dict(critic_state)

        # Initialize optimizers
        model._optimizer_generator = torch.optim.Adam(
            params=model._generator.parameters(),
            lr=model._generator_lr,
            betas=(0.5, 0.9),
            eps=1e-8,
            weight_decay=model._generator_decay
        )
        
        model._optimizer_critic = torch.optim.Adam(
            params=model._critic.parameters(),
            lr=model._critic_lr,
            betas=(0.5, 0.9),
            eps=1e-8,
            weight_decay=model._critic_decay
        )

        model._optimizer_generator.load_state_dict(additional_data["optimizer_generator_state"])
        model._optimizer_critic.load_state_dict(additional_data["optimizer_critic_state"])

        model._losses = additional_data["losses"]

        return model
    
    def plot_losses(self):
        losses = pd.DataFrame(self._losses)
        f, ax = plt.subplots(figsize=(8,6))
        ax.plot(losses["epoch"], losses["generator_loss"])
        ax.plot(losses["epoch"], losses["critic_loss"])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(ax.get_lines(), ['Generator Loss', 'Critic Loss'], title="Losses", bbox_to_anchor=(1,1))
        plt.show()
    
    def sample(self, count: int, cols: list[int] | None = None, batch_size: int = 500):
        self._generator.eval()

        if cols is not None:
            assert len([col for col in cols if col < 0 or col >= self._cond_dim]) == 0, "Col index out of bounds"

        # Get col probs
        pmfs = []
        pmfs_padded = None
        col_starts = []
        max_category_count = 0

        if cols is None:
            col_dim = 0
            for col in self._transformer.category_counts():
                category_freqs = []
                total = 0
                col_starts.append(col_dim)
                col_dim += len(col)
                max_category_count = max(len(col), max_category_count)
                for _, value in col.items():
                    category_freqs.append(value)
                    total += value
                # Normalize
                for i in range(len(category_freqs)):
                    category_freqs[i] /= total
                pmfs.append(category_freqs)

            col_starts = np.array(col_starts)

            pmfs_padded = np.zeros((len(pmfs), max_category_count))
            for i, pmf in enumerate(pmfs):
                pmfs_padded[i, :len(pmf)] = pmf

        # Condvec sample helper
        def _sample_condvec(batch_size: int, pmfs: np.ndarray, col_starts: np.ndarray):
            # Select discrete columns
            selected_column_idxs = np.random.choice(np.arange(len(pmfs)), batch_size)

            # Get PMF of selected discrete column
            cumsum_pmfs = np.cumsum(pmfs[selected_column_idxs], axis=1)

            # Select category based on pmf
            random = np.expand_dims(np.random.rand(cumsum_pmfs.shape[0]), axis=1)
            category_idxs = np.argmax(cumsum_pmfs > random, axis=1)

            column_idxs = col_starts[selected_column_idxs] + category_idxs

            cond_vector = np.zeros((batch_size, self._cond_dim), dtype=np.float32)
            cond_vector[np.arange(batch_size), column_idxs] = 1.0

            return cond_vector

        results = []

        with torch.no_grad():
            while count:
                num = min(batch_size, count)

                cond = None
                if cols is None:
                    cond = torch.from_numpy(_sample_condvec(batch_size=num, pmfs=pmfs_padded, col_starts=col_starts)).to(device=self._device)
                else:
                    cond = torch.zeros(num, self._cond_dim).to(device=self._device)
                    cond[:, cols] = 1.0

                result = self._generator(torch.randn(num, self._generator_input_dim).to(self._device), cond).cpu().numpy()

                result = np.round(result, decimals=3)
                inverse = self._transformer.inverse_transform(pd.DataFrame(result, columns=self._transformed_columns))

                results.append(inverse)

                count -= num

        self._generator.train()
        return pd.concat(results).reset_index(drop=True)


    def _critic_loss(self, real_datas: torch.Tensor, fake_datas: torch.Tensor, y_real: torch.Tensor, y_fake: torch.Tensor, cond_vector: torch.Tensor, gp_lambda: float = 10):
        wgan_loss = torch.mean(y_fake) - torch.mean(y_real)
        gp_loss = self._gradient_penalty(real_datas=real_datas, fake_datas=fake_datas, cond_vector=cond_vector)
        return wgan_loss + gp_lambda * gp_loss

    def _gradient_penalty(self, real_datas: torch.Tensor, fake_datas: torch.Tensor, cond_vector: torch.Tensor):
        alpha = torch.rand(real_datas.size(0), 1, device=self._device)
        alpha = alpha.expand_as(real_datas)
        
        # linear interpolation
        interpolated = alpha * real_datas + (1 - alpha) * fake_datas
        y_interpolated = self._critic(interpolated, cond_vector)

        # Calculate gradient on interpolated input againts interpolated critic output
        gradients = torch.autograd.grad(
            outputs=y_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(y_interpolated, device=self._device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradient_norm = gradients.norm(2, dim=1)
        return torch.mean((gradient_norm - 1) ** 2)

    def _generator_loss(self, fake_datas: torch.Tensor, y_fake: torch.Tensor, cond_vector: torch.Tensor, mask_vectors: torch.Tensor, col_idx):
        wgan_loss = -torch.mean(y_fake)
        cond_loss = self._cond_loss(fake_datas=fake_datas, cond_vector=cond_vector, mask_vectors=mask_vectors, col_idx=col_idx)

        return wgan_loss + cond_loss

    def _cond_loss(self, fake_datas: torch.Tensor, cond_vector: torch.Tensor, mask_vectors: torch.Tensor, col_idx):
        losses = []
        # Initial column offset
        cond_offset = 0
        offset = self._numerical_dim
        for col_info in self._generator.discrete_cols:
            col_dim = len(col_info.category_counts)

            target_mask = cond_vector[:, cond_offset:cond_offset + col_dim]
            generated_mask = fake_datas[:, offset:offset + col_dim]

            target_indices = torch.argmax(target_mask, dim=1)
            
            ce_loss = F.cross_entropy(generated_mask, target_indices, reduction='none')
            losses.append(ce_loss)

            cond_offset += col_dim
            offset += col_dim

        stacked_losses = torch.stack(losses, dim=1)
        masked_losses = stacked_losses * mask_vectors

        return masked_losses.sum() / fake_datas.size(0)