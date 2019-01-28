import heapq
import numpy as np


class Scheduler:
    def __init__(self, resource_ids, policy, stub, get_num_epochs_to_run):
        # List of resource types.
        self.resource_ids = resource_ids
        # Policy instance.
        self.policy = policy
        # Worker stub instance.
        self.stub = stub
        # get_num_epochs_to_run function pointer.
        self.get_num_epochs_to_run = get_num_epochs_to_run

        # Throughputs for all current incomplete applications.
        self.throughputs = {}
        # Allocations for all current incomplete applications.
        self.allocation = {}
        # Epochs run on each resource_id, for all current incomplete applications.
        self.run_so_far = {}
        # priority_queue for each resource_id.
        self.index = {}
        for resource_id in resource_ids:
            self.index[resource_id] = []

        self.last_app_id_assigned = 0

    def _get_allocation(self):
        def flatten(d):
            app_ids = list(d.keys())
            resource_ids = list(d[app_ids[0]].keys())
            m = []
            for app_id in app_ids:
                m_row = []
                for resource_id in resource_ids:
                    m_row.append(d[app_id][resource_id])
                m.append(m_row)
            return np.array(m), (app_ids, resource_ids)

        def unflatten(m, index):
            (app_ids, resource_ids) = index
            d = {}
            for i in range(len(app_ids)):
                d[app_ids[i]] = {}
                for j in range(len(resource_ids)):
                    d[app_ids[i]][resource_ids[j]] = m[i][j]
            return d

        flattened_throughputs, index = flatten(self.throughputs)
        flattened_allocation = self.policy.get_allocation(flattened_throughputs)
        return unflatten(flattened_allocation, index)

    def add_new_job(self, throughputs):
        # Application is a collection of throughputs for each
        # resource_id. (right now, not considering app packing)

        # Public-facing API call to add a new job, updates the
        # internal allocation of resources to jobs.
        # An allocation is of the form {application: <fraction
        # of allocations on different resources.>}. Some scheduler
        # mechanism needs to ensure that each application receives
        # this fraction correctly.
        app_id = self.last_app_id_assigned
        self.last_app_id_assigned += 1
        self.throughputs[app_id] = throughputs
        self.allocation = self._get_allocation()
        self.run_so_far[app_id] = {}
        for resource_id in self.resource_ids:
            self.run_so_far[app_id][resource_id] = 0
            # Entries in the index are sorted by fraction_run/fraction_allocated,
            # then number of epochs run, then app_id.
            heapq.heappush(self.index[resource_id],
                           [0.0, 0, app_id])
        return app_id

    def remove_old_job(self, app_id):
        # Public-facing API call to remove a completed job, updates
        # the internal allocation of resources to jobs.
        del self.throughputs[app_id]
        self.allocation = self._get_allocation()
        del self.run_so_far[app_id]
        self.remove_from_index_and_update(app_id)

    def remove_from_index_and_update(self, old_app_id):
        for resource_id in self.resource_ids:
            for i in range(len(self.index[resource_id])):
                if self.index[resource_id][i][2] == old_app_id:
                    break
            self.index[resource_id].pop(i)
            heapq.heapify(self.index[resource_id])

    def add_to_index_and_update(self, new_app_id):
        # Re-sort keys given that all fractions have decreased but one.
        # TODO: Can optimize this.
        fractions = {}
        tot_epochs_run = {}
        for app_id in self.run_so_far:
            fractions[app_id] = {}
            tot_epochs_run[app_id] = 0
        for resource_id in self.resource_ids:
            for app_id in self.run_so_far:
                tot_epochs_run[app_id] += \
                    self.run_so_far[app_id][resource_id]
        for resouce_type in self.resource_ids:
            for app_id in self.run_so_far:
                if tot_epochs_run[app_id] == 0:
                    fractions[app_id][resource_id] = 0.0
                else:
                    fractions[app_id][resource_id] = \
                        self.run_so_far[app_id][resource_id] / tot_epochs_run[app_id]
            self.index[resource_id].append([0.0, 0, new_app_id])
            for i in range(len(self.index[resource_id])):
                [_, _, app_id] = self.index[resource_id][i]
                self.index[resource_id][i][0] = fractions[app_id][resource_id] / \
                    self.allocation[app_id][resource_id]
                self.index[resource_id][i][1] = self.run_so_far[app_id][resource_id]
            heapq.heapify(self.index[resource_id])

    def schedule(self, resource_id):
        # Schedules the _inactive_ application most in need of the passed-in
        # resource_id (that is, the resource with the lowest
        # fraction_run/fraction_allocated ratio).

        # Scheduler holds two internal data structures,
        # {(application, resource_id): num_epochs_run_on_resource}
        # & {(application, resource_id): allocation_fraction}.
        # As an algorithmic optimization, might be good to maintain
        # a heap of all currently inactive applications for each
        # resource, sorted by fraction_run/fraction_allocated ratio.

        # Get the app_id for this resource_id with minimum
        # fraction_run/fraction_allocated.
        [_, _, app_id] = self.index[resource_id][0]
        self.remove_from_index_and_update(app_id)

        # Number of epochs to run the application on needs to be
        # determined.
        num_epochs = self.get_num_epochs_to_run(app_id,
                                                resource_id)
        self.stub.run_application(app_id, resource_id,
                                  num_epochs)
        return app_id, num_epochs

    def schedule_callback(self, app_id, resource_id, num_epochs):
        # Now, we can update the data structures to reflect the
        # fact that active_application run on a particular resource_
        # type for a certain num_epochs.
        self.run_so_far[app_id][resource_id] += num_epochs

        self.add_to_index_and_update(app_id)
