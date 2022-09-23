< script >
    require(
    ['base/js/namespace', 'jquery'],
    function (jupyter, $) {
        $(jupyter.events).on("kernel_ready.Kernel", function () {
            jupyter.actions.call('jupyter-notebook:run-all-cells-below');
        });
        }
    );
</script >