from app.usage import parse_gpu, parse_pids, parse_size, parse_stats


def test_parse_docker_stats_line() -> None:
    found = parse_stats("aiserver-pkg_x\t1.27%\t2.602GiB / 62.58GiB\n")
    assert found["aiserver-pkg_x"] == (1.27, round(2.602 * 1024**3))
    assert parse_size("343.2MiB") == round(343.2 * 1024**2)


def test_gpu_bytes_follow_container_pids() -> None:
    gpu = parse_gpu("111, 100\n222, 2048\n")
    pids = parse_pids("PID\n222\n333\n")
    assert gpu[222] == 2048 * 1024 * 1024
    assert sum(gpu.get(pid, 0) for pid in pids) == 2048 * 1024 * 1024
