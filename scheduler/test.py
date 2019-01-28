import numpy as np
import scheduler

class TestPolicy:
    def get_allocation(self, throughputs):
        (m, n) = throughputs.shape
        return np.full((m, n), 1.0 / n)

class TestStub:
    def run_application(self, command, app_id, resource_id,
                        num_epochs):
        print("Running application_%d on %s for %d epochs: %s" %
            (app_id, resource_id, num_epochs, command))

def get_num_epochs_to_run(app_id, resource_id):
    return 1

def test():
    resource_ids = ["v100"]
    num_applications = 10
    run_so_far = {}
    s = scheduler.Scheduler(resource_ids, TestPolicy(), TestStub(),
                            get_num_epochs_to_run)
    for j in range(num_applications):
        s.add_new_job({resource_id: 10 for resource_id in resource_ids},
                      "cmd%d" % j)
    for i in range(100):
        app_id, num_epochs = s.schedule(resource_ids[0])
        s.schedule_callback(app_id, resource_ids[0], num_epochs)
        if app_id not in run_so_far:
            run_so_far[app_id] = 0
        run_so_far[app_id] += num_epochs
        if i > 70 and app_id > 1 and app_id < 10:
            s.remove_old_job(app_id)
        if i == 90:
            s.add_new_job({resource_id: 10 for resource_id in resource_ids},
                          "cmd10")
    print()
    print("Number of epochs run:", run_so_far)


if __name__ == '__main__':
    test()
