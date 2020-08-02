package filter

import (
	"github.com/naili-xing/singa_auto_scheduler/pkg/sascheduler/collection"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"sync"
)

const Workers int = 4

func CollectMaxValue(
	value string,
	state *framework.CycleState,
	nodes []*v1.Node,
	filteredNodesStatuses framework.NodeToStatusMap) *framework.Status {

	Max := collection.Data{Value: 0}
	for _, n := range nodes {
		if filteredNodesStatuses[n.GetName()].IsSuccess() {
			if collection.StrToInt64(n.Labels["samonitor/"+value]) > Max.Value {
				Max.Value = collection.StrToInt64(n.Labels["samonitor/MaxFreeMemory"])
			}
		}
	}
	if Max.Value == 0 {
		return framework.NewStatus(framework.Error, " The max "+value+" of the nodes is 0")
	}
	state.Lock()
	state.Write(framework.StateKey("Max"+value), &Max)
	defer state.Unlock()
	return framework.NewStatus(framework.Success, "")
}

func ParallelCollection(
	workers int,
	state *framework.CycleState,
	nodes []*v1.Node,
	filteredNodesStatuses framework.NodeToStatusMap) *framework.Status {

	var (
		stop <-chan struct{}
		mx   sync.RWMutex
		msg  = ""
	)
	pieces := len(collection.Sum)
	toProcess := make(chan string, pieces)
	for _, v := range collection.Sum {
		toProcess <- v
	}
	close(toProcess)
	if pieces < workers {
		workers = pieces
	}
	wg := sync.WaitGroup{}
	wg.Add(workers)
	for i := 0; i < workers; i++ {
		go func() {
			for value := range toProcess {
				select {
				case <-stop:
					return
				default:
					if re := CollectMaxValue(value, state, nodes, filteredNodesStatuses); !re.IsSuccess() {
						klog.V(3).Infof(re.Message())
						mx.Lock()
						msg += re.Message()
						mx.Unlock()
					}
				}
			}
			wg.Done()
		}()
	}
	wg.Wait()
	if msg != "" {
		return framework.NewStatus(framework.Error, msg)
	}
	return framework.NewStatus(framework.Success, "")
}
