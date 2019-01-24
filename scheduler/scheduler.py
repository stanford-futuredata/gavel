import heapq


class Scheduler:
    def __init__(self, resource_types):
        # Throughputs for all current incomplete applications.
        self.throughputs = {}
        # Allocations for all current incomplete applications.
        self.allocation = {}
        # Epochs run on each resource_type, for all current incomplete
        # applications.
        self.run_so_far = {}
        # priority_queue for each resource_type.
        self.index = {}
        self.resource_types = resource_types
        for resource_type in resource_types:
            self.index[resource_type] = []

        self.last_application_id_assigned = 0

    def _get_allocation(self):
        def flatten(d):
            application_ids = d.keys()
            resource_types = d[application_ids[0]].keys()
            m = []
            for application_id in application_ids:
                m_row = []
                for resource_type in resource_types:
                    m_row.append(d[application_id][resource_type])
                m.append(m_row)
            return np.array(m), (application_ids, resource_types)

        def unflatten(m, (application_ids, resource_types)):
            d = {}
            for i in range(len(application_ids)):
                d[application_ids[i]] = {}
                for j in range(len(resource_types)):
                    d[application_ids[i]][resource_types[j]] = m[i][j]
            return d

        flattened_throughputs, index = flatten(self.throughputs)
        flattened_allocation = policy.get_allocation(flattened_throughputs)
        return unflatten(flattened_allocation, index)

    def add_new_job(self, throughputs):
        # Application is a collection of throughputs for each
        # resource_type. (right now, not considering app packing)

        # Public-facing API call to add a new job, updates the
        # internal allocation of resources to jobs.
        # An allocation is of the form {application: <fraction
        # of allocations on different resource_types.>}. Some scheduler
        # mechanism needs to ensure that each application receives
        # this fraction correctly.
        application_id = self.last_application_id_assigned
        self.last_application_id_assigned += 1
        self.throughputs[application_id] = throughputs
        self.allocation = self._get_allocation()
        self.run_so_far[application_id] = {}
        for resource_type in self.resource_types:
            self.run_so_far[application_id][resource_type] = 0
            heapq.heappush(self.index[resource_type],
                           [1. / len(self.resource_types), application_id])
        return application_id

    def remove_old_job(self, application_id):
        # Public-facing API call to remove a completed job, updates
        # the internal allocation of resources to jobs.
        del self.throughputs[application_id]
        self.allocation = self._get_allocation()
        delete self.run_so_far[application_id]
        for resource_type in self.resource_types:
            for i in range(len(self.index[resource_type])):
                if self.index[resource_type][i][1] == application_id:
                    break
            self.index[resource_type].pop(i)
            heapq.heapify(self.index[resource_type])

    def remove_from_index_and_update(self, old_application_id):
        for resource_type in self.resource_types:
            for i in range(self.index[resource_type]):
                if self.index[resource_type][i][1] == old_application_id:
                    break
            self.index[resource_type].pop(i)
            heapq.heapify(self.index[resource_type])

    def add_to_index_and_update(self, new_application_id):
        # Re-sort keys given that all fractions have decreased but one.
        # TODO: Can optimize this.
        for resource_type in self.resource_types:
            fractions = {}
            tot_epochs_run = 0
            for application_id in self.run_so_far:
                tot_epochs_run += self.run_so_far[application_id][resource_type]
            for application_id in self.run_so_far:
                fractions[resource_type] = \
                    self.run_so_far[application_id][resource_type] / tot_epochs_run
            self.index[resource_type].append([0.0, new_application_id])
            for i in range(len(self.index[resource_type])):
                [_, application_id] = self.index[resource_type][i]
                self.index[resource_type][i][0] = fractions[resource_type] / \
                    allocation[application_id][resource_type]
            heapq.heapify(self.index[resource_type])

    def schedule(self, resource_type):
        # Schedules the _inactive_ application most in need of the passed-in
        # resource_type (that is, the resource with the lowest
        # fraction_run/fraction_allocated ratio).

        # Scheduler holds two internal data structures,
        # {(application, resource_type): num_epochs_run_on_resource}
        # & {(application, resource_type): allocation_fraction}.
        # As an algorithmic optimization, might be good to maintain
        # a heap of all currently inactive applications for each
        # resource, sorted by fraction_run/fraction_allocated ratio.

        # Get the application_id for this resource_type with minimum
        # fraction_run/fraction_allocated.
        [_, application_id] = self.index[resource_type][0]
        self.remove_from_index_and_update(application_id)

        # Number of epochs to run the application on needs to be
        # determined.
        num_epochs = self.get_num_epochs_to_run(application_id,
                                                resource_type)
        run_application(application_id, resource_type,
                        num_epochs)

    def schedule_callback(self, application_id, resource_type, num_epochs):
        # Now, we can update the data structures to reflect the
        # fact that active_application run on a particular resource_
        # type for a certain num_epochs.
        self.run_so_far[application_id][resource_type] += num_epochs

        self.add_to_index_and_update(application_id)
