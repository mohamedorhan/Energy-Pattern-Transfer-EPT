# energy_pattern_transfer_simulation.py
"""
Energy Pattern Transfer (EPT) - Comprehensive Simulation Package
Author: Mohammed Orhan Zeineli
ORCID: 0009-0008-1139-8102
Email: mohamedorhanzeinel@gmail.com
GitHub: https://github.com/mohamedorhan/Energy-Pattern-Transfer-EPT

This module provides comprehensive simulations for the Energy Pattern Transfer (EPT) paradigm
as described in the research paper "Energy Pattern Transfer: A Third Paradigm for Electric Power Delivery"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

class EPTSimulator:
    """
    Main simulator class for Energy Pattern Transfer analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize EPT simulator with configuration parameters
        
        Parameters:
        config: Dictionary containing simulation parameters
        """
        self.default_config = {
            # System parameters
            'V_bus': 400.0,           # DC bus voltage [V]
            'R_line': 0.2,            # Line resistance [Ω]
            'eta_ch': 0.98,           # Charge efficiency
            'eta_dis': 0.98,          # Discharge efficiency
            
            # Storage parameters
            'E_min': 1e5,             # Minimum storage energy [J]
            'E_max': 1.3e5,           # Maximum storage energy [J]
            'E_ref_factor': 0.5,      # Reference SOC factor
            'K_soc': 1e-6,            # SOC feedback gain
            
            # Pattern parameters
            'K_fourier': 10,          # Fourier series order
            'delta_T': 1.0,           # Pattern update period [s]
            'prediction_horizon': 10.0,  # Prediction horizon [s]
            
            # Simulation parameters
            'T_total': 60.0,          # Total simulation time [s]
            'dt': 1e-3,               # Time step [s]
            
            # Load profile parameters
            'P_avg': 2000.0,          # Average power [W]
            'load_components': [      # Load frequency components
                {'amp': 800, 'freq': 0.5, 'phase': 0.0},
                {'amp': 700, 'freq': 2.0, 'phase': 0.5},
                {'amp': 500, 'freq': 0.2, 'phase': 0.0, 'burst': True}
            ]
        }
        
        self.config = {**self.default_config, **(config or {})}
        self.results = {}
        
    def generate_load_profile(self, t: np.ndarray) -> np.ndarray:
        """
        Generate complex load profile with multiple frequency components and bursts
        
        Parameters:
        t: Time array
        
        Returns:
        Load power profile
        """
        P_load = self.config['P_avg'] * np.ones_like(t)
        
        for component in self.config['load_components']:
            if component.get('burst', False):
                # Burst component (rectangular pulses)
                burst_signal = (np.sin(2 * np.pi * component['freq'] * t) > 0.8).astype(float)
                P_load += component['amp'] * burst_signal
            else:
                # Sinusoidal component
                P_load += component['amp'] * np.sin(
                    2 * np.pi * component['freq'] * t + component['phase']
                )
        
        return P_load
    
    def compute_spectral_pattern(self, P_load: np.ndarray, t: np.ndarray) -> Dict:
        """
        Compute spectral pattern using Fourier series decomposition
        
        Parameters:
        P_load: Load power profile
        t: Time array
        
        Returns:
        Dictionary containing pattern coefficients and reconstructed signal
        """
        K = self.config['K_fourier']
        T_window = self.config['prediction_horizon']
        
        # Use last window of data for pattern computation
        window_size = int(T_window / self.config['dt'])
        if len(P_load) < window_size:
            P_window = P_load
            t_window = t
        else:
            P_window = P_load[-window_size:]
            t_window = t[-window_size:]
        
        # Remove DC component for Fourier analysis
        P_dc = np.mean(P_window)
        P_ac = P_window - P_dc
        
        # Compute Fourier coefficients
        coefficients = {'a0': P_dc}
        omega_0 = 2 * np.pi / T_window
        
        for k in range(1, K + 1):
            omega_k = k * omega_0
            cos_component = np.cos(omega_k * t_window)
            sin_component = np.sin(omega_k * t_window)
            
            a_k = (2 / len(t_window)) * np.sum(P_ac * cos_component)
            b_k = (2 / len(t_window)) * np.sum(P_ac * sin_component)
            
            coefficients[f'a{k}'] = a_k
            coefficients[f'b{k}'] = b_k
        
        # Reconstruct signal from pattern
        P_reconstructed = coefficients['a0'] * np.ones_like(t_window)
        for k in range(1, K + 1):
            omega_k = k * omega_0
            P_reconstructed += (coefficients[f'a{k}'] * np.cos(omega_k * t_window) +
                              coefficients[f'b{k}'] * np.sin(omega_k * t_window))
        
        return {
            'coefficients': coefficients,
            'P_reconstructed': P_reconstructed,
            't_window': t_window,
            'original_signal': P_window
        }
    
    def ept_control_policy(self, E: float, P_avg: float) -> float:
        """
        Implement EPT near-constant current control policy
        
        Parameters:
        E: Current storage energy
        P_avg: Average power
        
        Returns:
        Controlled current
        """
        E_ref = (self.config['E_min'] + self.config['E_max']) * self.config['E_ref_factor']
        I_desired = (P_avg / self.config['V_bus'] + 
                    self.config['K_soc'] * (E_ref - E))
        
        # Clip to reasonable limits
        I_min = 0.0
        I_max = 15.0
        return np.clip(I_desired, I_min, I_max)
    
    def run_simulation(self) -> Dict:
        """
        Run complete EPT simulation
        
        Returns:
        Dictionary containing all simulation results
        """
        # Time array
        t = np.arange(0, self.config['T_total'], self.config['dt'])
        
        # Generate load profile
        P_load = self.generate_load_profile(t)
        
        # Conventional system baseline
        I_trad = P_load / self.config['V_bus']
        
        # EPT simulation
        E = (self.config['E_min'] + self.config['E_max']) * self.config['E_ref_factor']
        E_history = []
        I_ept = np.zeros_like(t)
        P_out = np.zeros_like(t)
        
        for i, t_i in enumerate(t):
            # Control policy
            I_ept[i] = self.ept_control_policy(E, self.config['P_avg'])
            P_out[i] = P_load[i]  # Ideal tracking assumption
            
            # Storage dynamics
            dE_dt = (self.config['eta_ch'] * self.config['V_bus'] * I_ept[i] - 
                    P_out[i] / self.config['eta_dis'])
            E = np.clip(E + dE_dt * self.config['dt'], 
                       self.config['E_min'], self.config['E_max'])
            E_history.append(E)
        
        E_history = np.array(E_history)
        
        # Compute spectral pattern
        pattern_results = self.compute_spectral_pattern(P_load, t)
        
        # Performance metrics
        L_trad = np.sum(I_trad**2) * self.config['R_line'] * self.config['dt']
        L_ept = np.sum(I_ept**2) * self.config['R_line'] * self.config['dt']
        
        I_trad_rms = np.sqrt(np.mean(I_trad**2))
        I_ept_rms = np.sqrt(np.mean(I_ept**2))
        
        sigma_I_sq = np.mean((I_trad - self.config['P_avg']/self.config['V_bus'])**2)
        theoretical_ratio = ((self.config['P_avg']/self.config['V_bus'])**2 / 
                           ((self.config['P_avg']/self.config['V_bus'])**2 + sigma_I_sq))
        
        soc_utilization = ((np.max(E_history) - np.min(E_history)) / 
                          (self.config['E_max'] - self.config['E_min']))
        
        # Pattern accuracy metrics
        pattern_error = np.mean(np.abs(pattern_results['P_reconstructed'] - 
                                     pattern_results['original_signal']))
        pattern_variance = np.var(pattern_results['P_reconstructed'])
        
        # Store results
        self.results = {
            'time': t,
            'P_load': P_load,
            'I_trad': I_trad,
            'I_ept': I_ept,
            'E_history': E_history,
            'pattern': pattern_results,
            'metrics': {
                'conventional_loss': L_trad,
                'ept_loss': L_ept,
                'loss_ratio': L_ept / L_trad,
                'efficiency_improvement': 1 - (L_ept / L_trad),
                'I_trad_rms': I_trad_rms,
                'I_ept_rms': I_ept_rms,
                'current_variance': sigma_I_sq,
                'theoretical_bound': theoretical_ratio,
                'soc_utilization': soc_utilization,
                'pattern_error': pattern_error,
                'pattern_variance': pattern_variance
            },
            'config': self.config
        }
        
        return self.results
    
    def generate_comprehensive_report(self):
        """Generate comprehensive performance report"""
        if not self.results:
            self.run_simulation()
        
        metrics = self.results['metrics']
        
        print("=" * 60)
        print("ENERGY PATTERN TRANSFER - COMPREHENSIVE SIMULATION REPORT")
        print("=" * 60)
        print(f"Simulation Duration: {self.config['T_total']} s")
        print(f"Time Step: {self.config['dt']} s")
        print(f"DC Bus Voltage: {self.config['V_bus']} V")
        print(f"Line Resistance: {self.config['R_line']} Ω")
        print()
        
        print("PERFORMANCE METRICS:")
        print("-" * 40)
        print(f"Conventional System Loss: {metrics['conventional_loss']:.1f} J")
        print(f"EPT System Loss: {metrics['ept_loss']:.1f} J")
        print(f"Loss Reduction Ratio: {metrics['loss_ratio']:.3f}")
        print(f"Efficiency Improvement: {metrics['efficiency_improvement']:.1%}")
        print(f"Theoretical Bound: {metrics['theoretical_bound']:.3f}")
        print()
        
        print(f"Traditional RMS Current: {metrics['I_trad_rms']:.2f} A")
        print(f"EPT RMS Current: {metrics['I_ept_rms']:.2f} A")
        print(f"Current Variance Reduction: {metrics['current_variance']:.2f} A²")
        print(f"Storage Utilization: {metrics['soc_utilization']:.1%}")
        print()
        
        print("PATTERN ANALYSIS:")
        print("-" * 40)
        print(f"Fourier Order (K): {self.config['K_fourier']}")
        print(f"Pattern Update Period: {self.config['delta_T']} s")
        print(f"Pattern Reconstruction Error: {metrics['pattern_error']:.2f} W")
        print(f"Pattern Variance: {metrics['pattern_variance']:.2f} W²")
    
    def plot_comprehensive_analysis(self, save_path: Optional[str] = None):
        """
        Generate comprehensive analysis plots
        
        Parameters:
        save_path: Path to save plots (optional)
        """
        if not self.results:
            self.run_simulation()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Energy Pattern Transfer - Comprehensive Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Power profiles comparison
        axes[0, 0].plot(self.results['time'], self.results['P_load'], 
                       'b-', linewidth=2, label='Actual Load')
        axes[0, 0].plot(self.results['pattern']['t_window'], 
                       self.results['pattern']['P_reconstructed'], 
                       'r--', linewidth=2, label='Pattern Reconstruction')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Power (W)')
        axes[0, 0].set_title('Power Profile and Pattern Reconstruction')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Current comparison
        axes[0, 1].plot(self.results['time'], self.results['I_trad'], 
                       'r-', linewidth=2, label='Conventional')
        axes[0, 1].plot(self.results['time'], self.results['I_ept'], 
                       'g-', linewidth=2, label='EPT')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Current (A)')
        axes[0, 1].set_title('Trunk Current Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Storage dynamics
        soc = (self.results['E_history'] - self.config['E_min']) / \
              (self.config['E_max'] - self.config['E_min'])
        axes[0, 2].plot(self.results['time'], soc * 100, 'purple', linewidth=2)
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('State of Charge (%)')
        axes[0, 2].set_title('Storage SOC Dynamics')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim([0, 100])
        
        # Plot 4: Loss analysis
        loss_cumulative_trad = np.cumsum(self.results['I_trad']**2) * \
                              self.config['R_line'] * self.config['dt']
        loss_cumulative_ept = np.cumsum(self.results['I_ept']**2) * \
                             self.config['R_line'] * self.config['dt']
        
        axes[1, 0].plot(self.results['time'], loss_cumulative_trad, 
                       'r-', linewidth=2, label='Conventional')
        axes[1, 0].plot(self.results['time'], loss_cumulative_ept, 
                       'g-', linewidth=2, label='EPT')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Cumulative Loss (J)')
        axes[1, 0].set_title('Cumulative Energy Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Frequency spectrum
        f_trad, Pxx_trad = signal.welch(self.results['I_trad'], 
                                       fs=1/self.config['dt'], nperseg=1024)
        f_ept, Pxx_ept = signal.welch(self.results['I_ept'], 
                                     fs=1/self.config['dt'], nperseg=1024)
        
        axes[1, 1].semilogy(f_trad, Pxx_trad, 'r-', label='Conventional')
        axes[1, 1].semilogy(f_ept, Pxx_ept, 'g-', label='EPT')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('PSD [A²/Hz]')
        axes[1, 1].set_title('Current Power Spectral Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim([0, 10])
        
        # Plot 6: Performance summary
        metrics = self.results['metrics']
        performance_data = [
            metrics['conventional_loss'],
            metrics['ept_loss'],
            metrics['I_trad_rms'],
            metrics['I_ept_rms'],
            metrics['soc_utilization'] * 100
        ]
        labels = ['Loss (J)', 'Loss (J)', 'RMS I (A)', 'RMS I (A)', 'SOC Util (%)']
        categories = ['Conventional\nLoss', 'EPT\nLoss', 'Conventional\nRMS I', 
                     'EPT\nRMS I', 'Storage\nUtilization']
        
        colors = ['red', 'green', 'red', 'green', 'purple']
        bars = axes[1, 2].bar(range(len(performance_data)), performance_data, 
                             color=colors, alpha=0.7)
        axes[1, 2].set_xticks(range(len(performance_data)))
        axes[1, 2].set_xticklabels(categories, rotation=45)
        axes[1, 2].set_title('Performance Summary')
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, performance_data):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def sensitivity_analysis(self, parameter: str, values: List[float]) -> pd.DataFrame:
        """
        Perform sensitivity analysis on a specific parameter
        
        Parameters:
        parameter: Parameter name to analyze
        values: List of parameter values to test
        
        Returns:
        DataFrame with sensitivity results
        """
        results = []
        
        original_value = self.config[parameter]
        
        for value in values:
            self.config[parameter] = value
            self.run_simulation()
            
            result_row = {
                'parameter': parameter,
                'value': value,
                **self.results['metrics']
            }
            results.append(result_row)
        
        # Restore original value
        self.config[parameter] = original_value
        
        return pd.DataFrame(results)

def run_demonstration():
    """
    Run comprehensive demonstration of EPT simulation
    """
    print("Energy Pattern Transfer - Demonstration")
    print("=" * 50)
    
    # Create simulator instance
    simulator = EPTSimulator()
    
    # Run simulation
    print("Running EPT simulation...")
    results = simulator.run_simulation()
    
    # Generate report
    simulator.generate_comprehensive_report()
    
    # Create plots
    print("\nGenerating comprehensive analysis plots...")
    simulator.plot_comprehensive_analysis('ept_comprehensive_analysis.png')
    
    # Sensitivity analysis example
    print("\nPerforming sensitivity analysis...")
    k_values = [5, 10, 15, 20, 25]
    sensitivity_df = simulator.sensitivity_analysis('K_fourier', k_values)
    
    # Plot sensitivity results
    plt.figure(figsize=(10, 6))
    plt.plot(sensitivity_df['value'], sensitivity_df['pattern_error'], 'bo-', linewidth=2)
    plt.xlabel('Fourier Order (K)')
    plt.ylabel('Pattern Reconstruction Error (W)')
    plt.title('Sensitivity Analysis: Pattern Accuracy vs Fourier Order')
    plt.grid(True, alpha=0.3)
    plt.savefig('ept_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nDemonstration completed successfully!")
    print("Generated files:")
    print("- ept_comprehensive_analysis.png")
    print("- ept_sensitivity_analysis.png")
    
    return simulator, sensitivity_df

if __name__ == "__main__":
    # Run demonstration when script is executed directly
    simulator, sensitivity_results = run_demonstration()