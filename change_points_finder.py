"""Class using Neural SDEs to find change-point in multivariate time series"""
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import ruptures as rpt
from utils import moving_average
from sde_model import LatentSDE
import math


class SDEChangePointFinder:
    """Finds change-point in time series using SDEs"""
    def __init__(self, pred_CP, pred_CP_2, time_series, directory, name):
        self.name = name
        self.predicted_change_point = pred_CP #this is defined using the mean-change detection function
        self.predicted_change_point_2 = pred_CP_2 #This is defined by finding the argmax D_S - D_{S-1}
        
        self.dim = time_series.shape[-1]
        self.root_folder = directory
        if not os.path.exists(directory):
            os.mkdir(directory)
            
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._num_predictions = 32
        self._epochs = 15
        self._rolling_number = 7
        self._rolling_size = 16
        self._window_size = 50
        self._ts = torch.linspace(0, 1, self._window_size, device=self._device)
        self._windowise(time_series)
    
    def set_seeds(self, num):
        np.random.seed(num)
        torch.manual_seed(num)
    
    def find_change_point(self):
        """Predicts the change-point in the time series"""
        # fit SDE to data
        self._sde = self._load_latent_sde()
        self._predictions = (
        torch.empty_like(self._time_series)
        .unsqueeze(1)
        .repeat(1, self._num_predictions, 1)
        )
        self._window_time_series = torch.empty_like(self._time_series)
        for window_index in tqdm(self._window_indices, desc="(Predicting)"):
            window = self._windows[:, window_index]
            self._predictions[window_index : window_index + self._window_size] = self._predict(window)
            self._window_time_series[
                window_index : window_index + self._window_size
            ] = window[:, -self.dim :]
        
        # compute distances, which we call `likelihoods'
        differences = (
        self._time_series.unsqueeze(1).repeat(1, self._num_predictions, 1) - self._predictions
        )
        likelihoods = - torch.square(differences).sum(dim=-1).mean(dim=1)
        absolute_differences = likelihoods[1:]- likelihoods[:-1]
        absolute_differences = absolute_differences[:-self._rolling_number * self._rolling_size]
        self.predicted_change_point_2 = self._sudden_drop(absolute_differences.cpu())
        
        likelihoods = moving_average(likelihoods, window_size=2 * self._window_size + 1)
        self._likelihoods = likelihoods[:-self._rolling_number * self._rolling_size]
        self.predicted_change_point = self._sudden_drop(self._likelihoods.cpu())
        
        
    def plot_with_uncertainty_estimates(self, true_change_point=None, change_point_lst = None):
        """Plot time series, true change-point, and predicted change-point"""
        plt.title(self.name + " (Last Dimension)")
        time_series = self._time_series.cpu()
        t = np.arange(len(time_series))
        ymin, ymax = min(time_series[:, -1]), max(time_series[:, -1])
        plt.plot(t, time_series, ",", color="blue")
        if true_change_point is not None:
            plt.vlines(
                [true_change_point],
                ymin=ymin,
                ymax=ymax,
                color="red",
                label="true change-point",
            )
            
        predicted_change_point = np.mean(change_point_lst)
        
        sd = np.std(change_point_lst)
        
        if change_point_lst is not None:
            plt.vlines(
                [predicted_change_point],
                ymin=ymin,
                ymax=ymax,
                color="orange",
                label="predicted change-point",
            )
        
        if len(change_point_lst) > 1: 
            interval = [predicted_change_point - sd, predicted_change_point + sd]
            plt.axvline(interval[0], color='blue', linestyle='dashed', label='Mean - 1 Std Dev')
            plt.axvline(interval[1], color='green', linestyle='dashed', label='Mean + 1 Std Dev')
            # Shading between lines
            plt.fill_betweenx(y=[-2, 2], x1=interval[0], x2=interval[1], color='gray', alpha=0.3, label='1 Std Dev Range')
            
        plt.legend()
        plt.savefig(os.path.join(self.root_folder, "result.pdf"), dpi=300)
        plt.close()
        
        
        
    def _make_windows(self, time_series):
        """Create sub-windows of time_series"""

        stack = [
            torch.roll(time_series, shifts=k * self._rolling_size, dims=(0,))
            for k in range(1, self._rolling_number)
        ] + [time_series]

        time_series = torch.cat(stack, dim=-1)
        stack = [
            time_series[i : i + self._window_size]
            for i in range(len(time_series) - self._window_size)
        ]
        
        return torch.stack(stack, dim=1)
    
    def _windowise(self, time_series):
        """Normalise time series and convert it to sub-windows"""
        # convert time series to torch tensor
        time_series = torch.tensor(
            time_series, dtype=torch.float32, device=self._device
        )
        self._time_series = time_series[..., -self.dim:]
        # z-normalise time series
        time_series = (time_series - time_series.mean(dim=(0, 1))) / time_series.std(
            dim=(0, 1)
        )
        # generate windows
        if self.predicted_change_point == None:
            self._windows_A = self._make_windows(time_series[: len(time_series) // 2])
        else:
            self._windows_A = self._make_windows(time_series[: int(self.predicted_change_point)])
        self._windows = self._make_windows(time_series)
        self._window_indices = torch.arange(0, self._windows.shape[1], self._window_size)
        

    def _load_latent_sde(self):
        """Train an SDE (or load it if it already exists)"""
        sde = LatentSDE(
            self._windows_A.shape[-1],
            context_size=4,
            hidden_size=64,
            output_size=self.dim,
        ).to(self._device)
    
        # where the SDE model is saved
        model_path = os.path.join(self.root_folder, "sde_model.pt")
        # train the SDE
        if not os.path.exists(model_path):
            optimiser = torch.optim.Adam(params=sde.parameters(), lr=1e-3)
            batch_size = min(self._windows_A.shape[1], 128)
            epochs = tqdm(torch.arange(self._epochs + 1), desc=f"(Training SDE)")
    
            for epoch in epochs:
                batch_selection = torch.randperm(self._windows_A.shape[1])[:batch_size]
                windows_batch = self._windows_A[:, batch_selection]
            
                log_pxs, log_qp, _ = sde(xs=windows_batch, ts=self._ts, dt=1 / len(self._ts))
                loss = -log_pxs + log_qp
                loss.backward(), optimiser.step(), sde.zero_grad()
        
            torch.save({"model_state_dict": sde.state_dict()}, model_path)
        
        else:
            checkpoint = torch.load(model_path)
            sde.load_state_dict(checkpoint["model_state_dict"])
        
        return sde
    
    def _predict(self, window):
        """Compute how likely the window is, given the SDE"""
        window = window.unsqueeze(1).repeat(1, self._num_predictions, 1)
        return self._sde.sample(ts=self._ts, z0=window[0], dt=1 / len(self._ts))
    
    def _sudden_drop(self, array):
        """Compute where likelihoods suddenly drop"""
        model = "l2"
        print('rpt with one cp', rpt.Dynp(model=model).fit(array).predict(n_bkps=1))
        change_point = rpt.Dynp(model=model).fit(array).predict(n_bkps=1)[0]
        minimal_change_point_position = len(self._time_series) // 2
        if not minimal_change_point_position <= change_point:
            change_point = (
                minimal_change_point_position
                + rpt.Dynp(model=model)
                .fit(array[minimal_change_point_position:])
                .predict(n_bkps=1)[0]                                    
            )
    
        return change_point