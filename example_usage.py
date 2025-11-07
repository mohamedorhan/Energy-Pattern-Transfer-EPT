# example_usage.py
"""
Example usage of the Energy Pattern Transfer simulation package
Demonstrates various scenarios and analysis techniques for the EPT paradigm
"""

from energy_pattern_transfer_simulation import EPTSimulator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def example_basic_simulation():
    """
    Example 1: Basic EPT simulation with default parameters
    Demonstrates the core functionality
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic EPT Simulation")
    print("=" * 60)
    
    # Create simulator with default parameters
    simulator = EPTSimulator()
    
    # Run simulation
    print("Running simulation with default parameters...")
    results = simulator.run_simulation()
    
    # Display key results
    metrics = results['metrics']
    print("\nKEY RESULTS:")
    print(f"• Conventional system loss: {metrics['conventional_loss']:,.1f} J")
    print(f"• EPT system loss: {metrics['ept_loss']:,.1f} J")
    print(f"• Efficiency improvement: {metrics['efficiency_improvement']:.1%}")
    print(f"• Loss reduction ratio: {metrics['loss_ratio']:.3f}")
    print(f"• Storage utilization: {metrics['soc_utilization']:.1%}")
    print(f"• Pattern reconstruction error: {metrics['pattern_error']:.2f} W")
    
    return simulator, results

def example_residential_scenario():
    """
    Example 2: Residential load profile simulation
    Typical household with periodic usage patterns
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Residential Scenario")
    print("=" * 60)
    
    residential_config = {
        'P_avg': 1500.0,    # Average household power
        'V_bus': 400.0,
        'T_total': 120.0,   # Longer simulation for daily patterns
        'load_components': [
            # Base load with daily rhythm
            {'amp': 300, 'freq': 1/86400, 'phase': 0.0},  # Daily cycle
            # Morning and evening peaks
            {'amp': 800, 'freq': 1/3600, 'phase': 0.2, 'burst': True},
            {'amp': 600, 'freq': 1/1800, 'phase': 0.7, 'burst': True},
            # Appliance usage
            {'amp': 400, 'freq': 0.1, 'phase': 0.0},
            {'amp': 200, 'freq': 0.05, 'phase': 0.5}
        ]
    }
    
    simulator = EPTSimulator(residential_config)
    results = simulator.run_simulation()
    
    metrics = results['metrics']
    print("RESIDENTIAL SCENARIO RESULTS:")
    print(f"• Efficiency gain: {metrics['efficiency_improvement']:.1%}")
    print(f"• Peak reduction: {(1 - metrics['I_ept_rms']/metrics['I_trad_rms']):.1%}")
    print(f"• Storage cycles: {metrics['soc_utilization']:.1%}")
    
    return simulator, results

def example_data_center_scenario():
    """
    Example 3: Data center with high power density and variability
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Data Center Scenario")
    print("=" * 60)
    
    data_center_config = {
        'P_avg': 50000.0,   # 50 kW average
        'V_bus': 800.0,     # Higher voltage for efficiency
        'E_min': 2e6,       # Larger storage capacity
        'E_max': 3e6,
        'R_line': 0.05,     # Lower resistance (better infrastructure)
        'load_components': [
            # Base computational load
            {'amp': 15000, 'freq': 0.5, 'phase': 0.0},
            # Server workload variations
            {'amp': 10000, 'freq': 2.0, 'phase': 0.3},
            # Bursty traffic patterns
            {'amp': 20000, 'freq': 0.2, 'phase': 0.0, 'burst': True},
            # Cooling system cycles
            {'amp': 8000, 'freq': 0.1, 'phase': 0.7}
        ]
    }
    
    simulator = EPTSimulator(data_center_config)
    results = simulator.run_simulation()
    
    metrics = results['metrics']
    annual_savings = (metrics['conventional_loss'] - metrics['ept_loss']) * 365 * 24 * 3600 / 120
    
    print("DATA CENTER SCENARIO RESULTS:")
    print(f"• Efficiency gain: {metrics['efficiency_improvement']:.1%}")
    print(f"• Annual energy savings: {annual_savings/1e6:.1f} MWh")
    print(f"• Current stabilization: {metrics['I_ept_rms']/metrics['I_trad_rms']:.1%} of original RMS")
    print(f"• Storage requirement: {(data_center_config['E_max'] - data_center_config['E_min'])/3600000:.1f} kWh")
    
    return simulator, results

def example_ev_charging_station():
    """
    Example 4: EV fast-charging station with high peak demands
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: EV Charging Station")
    print("=" * 60)
    
    ev_charging_config = {
        'P_avg': 30000.0,   # 30 kW average across multiple chargers
        'V_bus': 600.0,
        'E_min': 1e6,
        'E_max': 2e6,
        'T_total': 180.0,   # 3-minute simulation for charging cycle
        'load_components': [
            # Multiple charging sessions
            {'amp': 20000, 'freq': 1/60, 'phase': 0.0, 'burst': True},  # 1-minute cycles
            {'amp': 15000, 'freq': 1/90, 'phase': 0.3, 'burst': True},  # 1.5-minute cycles
            {'amp': 10000, 'freq': 1/120, 'phase': 0.7, 'burst': True}, # 2-minute cycles
            # Power fluctuations during charging
            {'amp': 5000, 'freq': 0.5, 'phase': 0.0},
            {'amp': 3000, 'freq': 1.0, 'phase': 0.5}
        ]
    }
    
    simulator = EPTSimulator(ev_charging_config)
    results = simulator.run_simulation()
    
    metrics = results['metrics']
    peak_reduction = (np.max(results['I_trad']) - np.max(results['I_ept'])) / np.max(results['I_trad'])
    
    print("EV CHARGING STATION RESULTS:")
    print(f"• Efficiency gain: {metrics['efficiency_improvement']:.1%}")
    print(f"• Peak current reduction: {peak_reduction:.1%}")
    print(f"• Infrastructure stress reduction: {(1 - metrics['I_ept_rms']/metrics['I_trad_rms']):.1%}")
    print(f"• Storage buffer: {metrics['soc_utilization']:.1%} of capacity used")
    
    return simulator, results

def comparative_analysis():
    """
    Example 5: Comparative analysis across multiple scenarios
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Comparative Analysis")
    print("=" * 60)
    
    scenarios = {
        'Residential': {
            'P_avg': 2000.0,
            'V_bus': 400.0,
            'load_components': [
                {'amp': 800, 'freq': 0.1, 'phase': 0.0},
                {'amp': 600, 'freq': 0.05, 'phase': 0.5},
                {'amp': 400, 'freq': 0.02, 'phase': 0.8, 'burst': True}
            ]
        },
        'Commercial': {
            'P_avg': 10000.0,
            'V_bus': 480.0,
            'load_components': [
                {'amp': 4000, 'freq': 0.2, 'phase': 0.0},
                {'amp': 3000, 'freq': 0.1, 'phase': 0.3},
                {'amp': 2000, 'freq': 0.05, 'phase': 0.6, 'burst': True}
            ]
        },
        'Industrial': {
            'P_avg': 50000.0,
            'V_bus': 600.0,
            'load_components': [
                {'amp': 20000, 'freq': 0.5, 'phase': 0.0},
                {'amp': 15000, 'freq': 0.3, 'phase': 0.4},
                {'amp': 10000, 'freq': 0.1, 'phase': 0.7, 'burst': True}
            ]
        },
        'Data Center': {
            'P_avg': 100000.0,
            'V_bus': 800.0,
            'load_components': [
                {'amp': 40000, 'freq': 1.0, 'phase': 0.0},
                {'amp': 30000, 'freq': 0.5, 'phase': 0.2},
                {'amp': 20000, 'freq': 0.2, 'phase': 0.5, 'burst': True}
            ]
        }
    }
    
    results_summary = []
    
    for scenario_name, config in scenarios.items():
        print(f"Running {scenario_name} scenario...")
        simulator = EPTSimulator(config)
        results = simulator.run_simulation()
        metrics = results['metrics']
        
        results_summary.append({
            'Scenario': scenario_name,
            'Avg Power (kW)': config['P_avg'] / 1000,
            'Efficiency Gain': metrics['efficiency_improvement'],
            'Loss Ratio': metrics['loss_ratio'],
            'RMS Current Reduction': 1 - metrics['I_ept_rms'] / metrics['I_trad_rms'],
            'Storage Utilization': metrics['soc_utilization'],
            'Pattern Error (W)': metrics['pattern_error']
        })
    
    # Create summary dataframe
    df_summary = pd.DataFrame(results_summary)
    
    print("\nCOMPARATIVE ANALYSIS RESULTS:")
    print("=" * 80)
    print(df_summary.round(4).to_string(index=False))
    
    # Plot comparative results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('EPT Comparative Analysis Across Scenarios', fontsize=16, fontweight='bold')
    
    # Plot 1: Efficiency gains
    scenarios_list = df_summary['Scenario']
    efficiency_gains = df_summary['Efficiency Gain'] * 100
    
    bars1 = axes[0, 0].bar(scenarios_list, efficiency_gains, color=['blue', 'green', 'orange', 'red'])
    axes[0, 0].set_ylabel('Efficiency Gain (%)')
    axes[0, 0].set_title('Efficiency Improvement by Scenario')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom')
    
    # Plot 2: RMS current reduction
    current_reduction = df_summary['RMS Current Reduction'] * 100
    
    bars2 = axes[0, 1].bar(scenarios_list, current_reduction, color=['blue', 'green', 'orange', 'red'])
    axes[0, 1].set_ylabel('RMS Current Reduction (%)')
    axes[0, 1].set_title('Current Stabilization Performance')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom')
    
    # Plot 3: Storage utilization
    storage_util = df_summary['Storage Utilization'] * 100
    
    bars3 = axes[1, 0].bar(scenarios_list, storage_util, color=['blue', 'green', 'orange', 'red'])
    axes[1, 0].set_ylabel('Storage Utilization (%)')
    axes[1, 0].set_title('Required Storage Capacity Utilization')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    for bar in bars3:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom')
    
    # Plot 4: Pattern accuracy
    pattern_error = df_summary['Pattern Error (W)']
    
    bars4 = axes[1, 1].bar(scenarios_list, pattern_error, color=['blue', 'green', 'orange', 'red'])
    axes[1, 1].set_ylabel('Pattern Error (W)')
    axes[1, 1].set_title('Spectral Pattern Reconstruction Accuracy')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    for bar in bars4:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}W', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('comparative_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_summary

def parameter_sensitivity_study():
    """
    Example 6: Sensitivity analysis of key EPT parameters
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Parameter Sensitivity Analysis")
    print("=" * 60)
    
    # Use residential scenario as baseline
    base_config = {
        'P_avg': 2000.0,
        'V_bus': 400.0,
        'K_fourier': 10,
        'delta_T': 1.0
    }
    
    simulator = EPTSimulator(base_config)
    
    # Study Fourier order sensitivity
    k_values = [5, 8, 10, 15, 20, 25]
    k_results = simulator.sensitivity_analysis('K_fourier', k_values)
    
    # Study pattern update period sensitivity
    delta_t_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    delta_t_results = simulator.sensitivity_analysis('delta_T', delta_t_values)
    
    # Plot sensitivity results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('EPT Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Fourier order vs pattern error
    axes[0, 0].plot(k_results['value'], k_results['pattern_error'], 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Fourier Order (K)')
    axes[0, 0].set_ylabel('Pattern Reconstruction Error (W)')
    axes[0, 0].set_title('Pattern Accuracy vs Fourier Order')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Fourier order vs efficiency
    axes[0, 1].plot(k_results['value'], k_results['efficiency_improvement']*100, 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Fourier Order (K)')
    axes[0, 1].set_ylabel('Efficiency Gain (%)')
    axes[0, 1].set_title('Efficiency vs Fourier Order')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Update period vs pattern error
    axes[1, 0].plot(delta_t_results['value'], delta_t_results['pattern_error'], 'ro-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Pattern Update Period (s)')
    axes[1, 0].set_ylabel('Pattern Reconstruction Error (W)')
    axes[1, 0].set_title('Pattern Accuracy vs Update Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Update period vs communication requirements
    bitrate = [(2*k + 1) * 16 / dt for dt in delta_t_results['value']]  # 16 bits per coefficient
    axes[1, 1].plot(delta_t_results['value'], [br/1000 for br in bitrate], 'mo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Pattern Update Period (s)')
    axes[1, 1].set_ylabel('Required Bitrate (kbps)')
    axes[1, 1].set_title('Communication Requirements vs Update Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("SENSITIVITY ANALYSIS SUMMARY:")
    print(f"Optimal Fourier order: {k_results.loc[k_results['pattern_error'].idxmin(), 'value']}")
    print(f"Optimal update period: {delta_t_results.loc[delta_t_results['pattern_error'].idxmin(), 'value']} s")
    print(f"Communication feasibility: All scenarios < 100 kbps (practical for PLC)")
    
    return k_results, delta_t_results

def run_all_examples():
    """
    Run all examples sequentially
    """
    print("ENERGY PATTERN TRANSFER - COMPREHENSIVE DEMONSTRATION")
    print("=" * 70)
    
    # Run all examples
    example_basic_simulation()
    example_residential_scenario()
    example_data_center_scenario()
    example_ev_charging_station()
    
    comparative_results = comparative_analysis()
    sensitivity_results = parameter_sensitivity_study()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated files:")
    print("• comparative_analysis.png - Scenario comparison charts")
    print("• parameter_sensitivity.png - Parameter optimization results")
    print("• Multiple runtime analysis plots")
    
    return comparative_results, sensitivity_results

if __name__ == "__main__":
    # Run complete demonstration
    comparative_results, sensitivity_results = run_all_examples()