"""Physical action errors, weighted by valid commands; padded tails excluded."""
import numpy as np

def error_arrays(pred, gt, state, valid, fk, mean, std):
    err = pred - gt
    T = fk(pred[..., :7])
    G = fk(gt[..., :7])
    tcp = np.linalg.norm(T[..., :3, 3] - G[..., :3, 3], axis=-1) * 1000
    rot = T[..., :3, :3] @ np.swapaxes(G[..., :3, :3], -1, -2)
    ang = np.rad2deg(np.arccos(np.clip((np.trace(rot, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    return dict(joint_abs=np.abs(err[..., :7]), joint_sq=err[..., :7] ** 2, tcp_mm=tcp, orientation_deg=ang, gripper_abs=np.abs(err[..., 7]), gripper_wrong=(pred[..., 7] >= 0) != (gt[..., 7] >= 0), normalized_sq=((pred - gt) / std) ** 2)

def summarize_metrics(raw, fk, mean, std):
    gt = raw['gt']
    valid = raw['valid']
    state = raw['state']
    out = {}
    for (mode, pred) in [('teacher_forced', raw['tf']), ('generated', raw['gen']), ('hold_current_joints', np.repeat(state[:, None, :], 30, axis=1))]:
        e = error_arrays(pred, gt, state, valid, fk, mean, std)
        d = {}
        for horizon in [1, 16, 30]:
            mask = valid[:, :horizon]
            ja = e['joint_abs'][:, :horizon][mask]
            tc = e['tcp_mm'][:, :horizon][mask]
            d[f'first_{horizon}'] = {'joint_mae_rad': float(ja.mean()), 'joint_rmse_rad': float(np.sqrt(e['joint_sq'][:, :horizon][mask].mean())), 'joint_abs_p95_rad': float(np.quantile(ja, 0.95)), 'joint_mae_deg': float(np.rad2deg(ja.mean())), 'tcp_mean_mm': float(tc.mean()), 'tcp_p50_mm': float(np.median(tc)), 'tcp_p95_mm': float(np.quantile(tc, 0.95)), 'orientation_mean_deg': float(e['orientation_deg'][:, :horizon][mask].mean()), 'per_joint_mae_rad': ja.mean(0).tolist()}
            if mode != 'hold_current_joints':
                d[f'first_{horizon}'].update(normalized_action_mse=float(e['normalized_sq'][:, :horizon][mask].mean()), gripper_mae=float(e['gripper_abs'][:, :horizon][mask].mean()), gripper_sign_accuracy=float(1 - e['gripper_wrong'][:, :horizon][mask].mean()))
        d['horizon_joint_mae_rad'] = [float(e['joint_abs'][:, h][valid[:, h]].mean()) for h in range(30)]
        d['horizon_tcp_mean_mm'] = [float(e['tcp_mm'][:, h][valid[:, h]].mean()) for h in range(30)]
        out[mode] = d
    out['flow_velocity_mse'] = float(raw['flow_mse'].mean())
    out['zero_velocity_baseline_mse'] = float(raw['zero_flow_mse'].mean())
    out['flow_mse_per_dimension'] = raw['flow_mse_dim'].mean(0).tolist()
    out['one_step_clean_normalized_mse'] = float(raw['one_step_clean_mse'].mean())
    return out
