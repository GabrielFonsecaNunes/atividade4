function copyCode(containerId) {
    var container = document.getElementById(containerId);
    var textarea = container.querySelector('textarea');
    textarea.select();
    document.execCommand('copy');
    alert('Código copiado para a área de transferência!');
}
